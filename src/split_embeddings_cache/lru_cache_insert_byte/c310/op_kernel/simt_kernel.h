/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef SIMT_KERNEL_H
#define SIMT_KERNEL_H

#include <climits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "cache_constants.h"
#include "padded_row.h"
#include "weight_row.h"

using namespace AscendC;

// 与 v220 lru_cache_insert_byte_kernel.h 中 INT8 scale/bias 步长一致（非 FBGEMM cuda 8B）
constexpr int32_t kRecsdkScaleBiasBytes = 4;

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t DATA_ALIGN_BYTES = 32;

// 多 AI Core：SmallDataCompute 内用 GetBlockIdx/GetBlockNum 对 unique 前缀下标 n 分片（与 lru_cache_find_uncached SIMT 一致）。

namespace CumsumSimt {

template <typename KeyType, typename ValType>
__simt_callee__ inline void warp_sort_kv_optimized(KeyType& key, ValType& val, bool asc = true) {
    // 阶段 1: 构建双调序列
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // 获取伙伴线程的 key 和 val
            KeyType partner_key = asc_shfl_xor(key, j);
            ValType partner_val = asc_shfl_xor(val, j);
            
            // 计算是否需要交换
            bool high_pos = threadIdx.x & j;  // 是否在高位
            bool block_dir = (threadIdx.x & k) == 0;  // 块方向
            
            // 根据升序/降序调整块方向
            if (!asc) block_dir = !block_dir;
            
            // 根据位置和方向决定是否需要交换
            bool should_swap = false;
            
            if (block_dir) {
                // 升序方向
                if (high_pos) {
                    // 高位应取较大值
                    should_swap = (key < partner_key);
                } else {
                    // 低位应取较小值
                    should_swap = (partner_key < key);
                }
            } else {
                // 降序方向
                if (high_pos) {
                    // 高位应取较小值
                    should_swap = (partner_key < key);
                } else {
                    // 低位应取较大值
                    should_swap = (key < partner_key);
                }
            }
            
            // 如果需要交换，更新 key 和 val
            if (should_swap) {
                key = partner_key;
                val = partner_val;
            }
        }
    }
}


template <typename IndexT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
inline void SmallDataCompute(
    __gm__ uint8_t* weights,
    __gm__ int64_t* cache_hash_size_cumsum,
    __gm__ int32_t* cache_index_table_map,
    __gm__ int64_t* weights_offsets,
    __gm__ uint8_t* weights_tys,
    __gm__ int32_t* D_offsets,
    __gm__ int32_t* sorted_cache_sets,
    __gm__ IndexT* cache_set_sorted_indices,
    __gm__ int32_t* N_unique,
    __gm__ int64_t* lxu_cache_state,
    __gm__ uint8_t* lxu_cache_weights,
    int32_t lxu_cache_row_bytes,
    int32_t row_alignment,
    int64_t time_stamp,
    __gm__ int64_t* lru_state,
    bool gather_cache_stats,
    __gm__ int32_t* uvm_cache_stats,
    bool lock_cache_line,
    __gm__ int32_t* lxu_cache_locking_counter,
    int32_t num_cache_sets,
    int32_t num_ways,
    int32_t nbuf_cap)
{
    // 线程信息：与 CUDA lru_cache_insert 一致，整 warp 协作处理同一个 n；多核下 n 带 GetBlockIdx 偏移
    int32_t n_conflict_misses = 0;
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;
    // block 内 warp 个数；单 block 下等价于 CUDA 中 stride = gridDim.x * blockDim.y 里「每 block 的 y 向条数」
    const int32_t numWarpsPerBlock = MAX_THREADS_PER_BLOCK / WARP_SIZE;
    const int32_t bid = blockIdx.x;
    int32_t grid = gridDim.x;
    if (grid <= 0) {
        grid = 1;
    }

    int32_t numUnique = *N_unique;
    if (numUnique < 0) {
        numUnique = 0;
    }
    if (numUnique > nbuf_cap) {
        numUnique = nbuf_cap;
    }

    const int32_t waysCap = num_ways < WARP_SIZE ? num_ways : WARP_SIZE;

    // n = bid * numWarpsPerBlock + warpId，步进 grid * numWarpsPerBlock
    for (int32_t n = bid * numWarpsPerBlock + warpId; n < numUnique; n += grid * numWarpsPerBlock) {
      // check if this warp is responsible for this whole segment.
      const bool segment_start =
          (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);
  
      if (!segment_start) {
        // don't have *warp* divergence since we launch full warps in blockDim.x,
        // so we can just exit this warp entirely.
        continue;
      }
      const int32_t cache_set = sorted_cache_sets[n];
      if (cache_set == num_cache_sets) {
        // ignore the already-existing elements
        continue;
      }

      const int64_t rowBase = static_cast<int64_t>(cache_set) * static_cast<int64_t>(num_ways);
      
      int32_t SL = 1;
      while (n + SL < numUnique && sorted_cache_sets[n + SL] == cache_set) {
        SL += 1;
      }
      int32_t n_inserted = 0; // also used as index to insert
  
      // now, we need to insert the (unique!) values in indices[n:n + SL] into
      // our slots.
      // 一路 cache way 对应 warp 内 lane（0..WARP_SIZE-1），与 CUDA threadIdx.x 在 x 维一致
      const int32_t slot = laneId;
      const int64_t slot_time =
          (laneId < num_ways) ? lru_state[rowBase + laneId] : static_cast<int64_t>(LLONG_MAX);
      int64_t costs = slot_time;
      int32_t slots = slot;
  
      // BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
      warp_sort_kv_optimized(costs, slots);
      const int32_t sorted_slot = slots;
      const int64_t sorted_lru_cost = costs;

      for (int32_t l = 0; l < min(SL, waysCap); ++l) {
        const int32_t insert_slot = asc_shfl(sorted_slot, l);
        if (lock_cache_line) {
          const int32_t count = lxu_cache_locking_counter[rowBase + insert_slot];
          if (count > 0) {
            continue; // cache slot is in use
          }
        }
        const int64_t insert_current_lru_cost = asc_shfl(sorted_lru_cost, l);
        if (insert_current_lru_cost == time_stamp) {
          break;
        }
        const int64_t insert_idx = static_cast<int64_t>(cache_set_sorted_indices[n + n_inserted]);
        const int32_t t_insert = cache_index_table_map[insert_idx];
        const int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
        const int64_t weights_offset_insert = weights_offsets[t_insert];
        const int32_t D_start_insert = D_offsets[t_insert];
        const int32_t D_end_insert = D_offsets[t_insert + 1];
        const int32_t D_insert = D_end_insert - D_start_insert;
  
        // ensure that threadIdx.x is the only thread reading/writing to
        // lxu_cache_state
        int64_t current_idx =
            laneId == 0 ? lxu_cache_state[rowBase + insert_slot] : 0;
        current_idx = asc_shfl(current_idx, 0);
  
        // not empty
        if (current_idx != fbgemm_compat::kCacheStateInvalid) {
          // evict: cache slot -> UVM（fbgemm_compat::WeightRow，与 FBGEMM weight_row.cuh uint8 同型拷贝一致）
          const int32_t t_current = cache_index_table_map[current_idx];
          const int64_t idx_current =
              current_idx - cache_hash_size_cumsum[t_current];
          const int64_t weights_offset_current = weights_offsets[t_current];
          const int32_t D_start_current = D_offsets[t_current];
          const int32_t D_end_current = D_offsets[t_current + 1];
          const int32_t D_current = D_end_current - D_start_current;
          const uint8_t w_ty_current = weights_tys[t_current];
          const int32_t d_bytes_current = fbgemm_compat::PaddedRowSizeBytes(
              D_current, w_ty_current, row_alignment, kRecsdkScaleBiasBytes);

          __gm__ uint8_t* uvm_row_current =
              weights + weights_offset_current + idx_current * static_cast<int64_t>(d_bytes_current);
          __gm__ uint8_t* cache_row_ptr = lxu_cache_weights +
              ((rowBase + static_cast<int64_t>(insert_slot)) *
                  static_cast<int64_t>(lxu_cache_row_bytes));

          fbgemm_compat::WeightRow<uint8_t, uint8_t, uint8_t> weight_row(
              uvm_row_current, cache_row_ptr, D_current, false, nullptr, 0ULL);
          weight_row.warp_evict_cache(d_bytes_current, WARP_SIZE, laneId);
        }

        const uint8_t w_ty_insert = weights_tys[t_insert];
        const int32_t d_bytes_insert = fbgemm_compat::PaddedRowSizeBytes(
            D_insert, w_ty_insert, row_alignment, kRecsdkScaleBiasBytes);
        if (d_bytes_insert > lxu_cache_row_bytes) {
          break;
        }

        __gm__ uint8_t* uvm_row_ins =
            weights + weights_offset_insert + idx_insert * static_cast<int64_t>(d_bytes_insert);
        __gm__ uint8_t* cache_dst = lxu_cache_weights +
            ((rowBase + static_cast<int64_t>(insert_slot)) *
                static_cast<int64_t>(lxu_cache_row_bytes));

        fbgemm_compat::WeightRow<uint8_t, uint8_t, uint8_t> weight_row_emb(
            uvm_row_ins, nullptr, D_insert, false, nullptr, 0ULL);
        weight_row_emb.warp_copy_to_cache(cache_dst, d_bytes_insert, WARP_SIZE, laneId);
  
        if (laneId == 0) {
          lxu_cache_state[rowBase + insert_slot] = insert_idx;
          lru_state[rowBase + insert_slot] = time_stamp;
          if (lock_cache_line) {
            lxu_cache_locking_counter[rowBase + insert_slot] += 1;
          }
        }
  
        n_inserted++;
      }
      n_conflict_misses += (SL - n_inserted);
    }
    // 每个 warp 的 lane0 上报本 warp 累计的冲突 miss（与 find_uncached 中 lane0 原子加一致）
    if (gather_cache_stats && n_conflict_misses > 0 && laneId == 0) {
        atomicAdd(
            &uvm_cache_stats[fbgemm_compat::UvmCacheStatsIndex::num_conflict_unique_misses],
            n_conflict_misses);
    }
}

}  // namespace CumsumSimt

#endif  // SIMT_KERNEL_H
