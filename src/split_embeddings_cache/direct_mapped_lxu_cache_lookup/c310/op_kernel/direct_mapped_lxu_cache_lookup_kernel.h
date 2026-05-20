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

#ifndef DIRECT_MAPPED_LXU_CACHE_LOOKUP_KERNEL_H
#define DIRECT_MAPPED_LXU_CACHE_LOOKUP_KERNEL_H

#include "simt_kernel.h"

struct Args {
    GM_ADDR linear_cache_indices;
    GM_ADDR lxu_cache_state;
    GM_ADDR uvm_cache_stats;
    GM_ADDR lxu_cache_locations;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

namespace DirectMappedLxuCacheLookup {

template <typename T>
class DirectMappedLxuCacheLookupKernel {
public:
    __aicore__ inline DirectMappedLxuCacheLookupKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        this->invalid_index = tilingData.invalid_index;
        this->gather_cache_status = tilingData.gather_cache_status;
        this->indices = tilingData.indices;
        this->slots = tilingData.slots;
        this->uvm_len = tilingData.uvm_len;

        linear_cache_indices_gm = reinterpret_cast<__gm__ T*>(args.linear_cache_indices);
        lxu_cache_state_gm = reinterpret_cast<__gm__ int64_t*>(args.lxu_cache_state);
        uvm_cache_stats_gm = reinterpret_cast<__gm__ int32_t*>(args.uvm_cache_stats);
        lxu_cache_locations_gm = reinterpret_cast<__gm__ T*>(args.lxu_cache_locations);
    }

    __aicore__ inline void Compute()
    {
        constexpr int32_t kThreadNum = 1024;
        AscendC::Simt::Dim3 dim(kThreadNum, 1, 1);
        asc_vf_call<DirectMappedLxuCacheLookupSimt::SimtDirectMappedLxuCacheLookupMultiThread<T>>(
            dim, this->linear_cache_indices_gm, this->lxu_cache_state_gm, this->invalid_index,
            this->gather_cache_status, this->uvm_cache_stats_gm, this->lxu_cache_locations_gm, this->indices,
            this->slots, this->uvm_len);
    }

private:
    int32_t indices;
    int32_t slots;
    int32_t uvm_len;
    int64_t invalid_index;
    bool gather_cache_status;
    __gm__ T* linear_cache_indices_gm;
    __gm__ int64_t* lxu_cache_state_gm;
    __gm__ int32_t* uvm_cache_stats_gm;
    __gm__ T* lxu_cache_locations_gm;
};
}  // namespace DirectMappedLxuCacheLookup

#endif
