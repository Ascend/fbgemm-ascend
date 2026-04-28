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

#ifndef LRU_CACHE_INSERT_BYTE_FBGEMM_WEIGHT_ROW_H
#define LRU_CACHE_INSERT_BYTE_FBGEMM_WEIGHT_ROW_H

#include <cstdint>
#include <type_traits>

#ifndef __aicore__
#define __aicore__
#endif

namespace fbgemm_compat {

/**
 * row_: UVM (embedding) row start; cache_row_: current cache slot row start.
 * dim_ matches FBGEMM field name: logical embedding dim D (bytes of payload
 * before int8 qparams tail); warp_evict_cache/warp_copy_to_cache use byte
 * length arguments for actual copy span (typically padded row bytes).
 */
struct WeightRowU8 {
    __gm__ uint8_t* row_;
    __gm__ uint8_t* cache_row_;
    int32_t dim_;

    __aicore__ inline WeightRowU8(__gm__ uint8_t* row, __gm__ uint8_t* cache_row, int32_t dim)
        : row_(row), cache_row_(cache_row), dim_(dim)
    {}

    /// API-compatible with FBGEMM 7-arg ctor; stochastic / philox ignored on device.
    __aicore__ inline WeightRowU8(
        __gm__ uint8_t* row,
        __gm__ uint8_t* cache_row,
        int32_t dim,
        bool /* stochastic_rounding */,
        const void* /* stochastic_philox_args */,
        uint64_t /* salt_value */)
        : row_(row), cache_row_(cache_row), dim_(dim)
    {}

    /// Cooperative copy: UVM row -> dst (cache line), striping by lane.
    __aicore__ inline void warp_copy_to_cache(
        __gm__ uint8_t* dst_row, int32_t dim_bytes, int32_t num_lanes, int32_t lane_id) const
    {
        for (int32_t d = lane_id; d < dim_bytes; d += num_lanes) {
            dst_row[d] = row_[d];
        }
        (void)dim_;
    }

    /// Cooperative copy: cache line -> UVM row (evict), striping by lane.
    __aicore__ inline void warp_evict_cache(int32_t dim_bytes, int32_t num_lanes, int32_t lane_id) const
    {
        for (int32_t d = lane_id; d < dim_bytes; d += num_lanes) {
            row_[d] = cache_row_[d];
        }
        (void)dim_;
    }
};

/// Template facade matching FBGEMM WeightRow<emb_t, cache_t, dst_t> name for call sites.
template <typename emb_t, typename cache_t, typename dst_t>
struct WeightRow {
    static_assert(std::is_same_v<emb_t, uint8_t> && std::is_same_v<cache_t, uint8_t> &&
            std::is_same_v<dst_t, uint8_t>,
        "RecSDK fbgemm_compat::WeightRow: only uint8 / uint8 / uint8 (insert_byte) is implemented");
    WeightRowU8 impl_;

    __aicore__ inline WeightRow(__gm__ emb_t* row, __gm__ cache_t* cache_row, int32_t dim)
        : impl_(reinterpret_cast<__gm__ uint8_t*>(row), reinterpret_cast<__gm__ uint8_t*>(cache_row), dim)
    {}

    __aicore__ inline WeightRow(
        __gm__ emb_t* row,
        __gm__ cache_t* cache_row,
        int32_t dim,
        bool stochastic_rounding,
        const void* philox_args,
        uint64_t salt)
        : impl_(reinterpret_cast<__gm__ uint8_t*>(row),
              reinterpret_cast<__gm__ uint8_t*>(cache_row),
              dim,
              stochastic_rounding,
              philox_args,
              salt)
    {}

    __aicore__ inline void warp_copy_to_cache(
        __gm__ cache_t* dst_row, int32_t dim_bytes, int32_t num_lanes, int32_t lane_id) const
    {
        impl_.warp_copy_to_cache(
            reinterpret_cast<__gm__ uint8_t*>(dst_row), dim_bytes, num_lanes, lane_id);
    }

    __aicore__ inline void warp_evict_cache(int32_t dim_bytes, int32_t num_lanes, int32_t lane_id) const
    {
        impl_.warp_evict_cache(dim_bytes, num_lanes, lane_id);
    }
};

}  // namespace fbgemm_compat

#endif
