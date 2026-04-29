set(ASCENDC_STAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/ascendc_opp)
set(ASCENDC_STAGE_SUBDIRS "")
set_property(GLOBAL PROPERTY FBGEMM_ASCEND_TARGETS "")

set(ASCENDC_A5_ONLY_OPS
    float_or_half_to_fused_nbit_rowwise
    block_bucketize_sparse_features
    expand_into_jagged_permute
    invert_permute
    init_address_lookup
    int_nbit_split_embedding_codegen_lookup_function
    run_length_encode
    pruned_hashmap_lookup
    lru_cache_find_uncached
    lru_cache_insert_byte
    group_index_select_dim0
    group_index_select_dim0_backward
)

set(ASCENDC_A3_OPS
    asynchronous_complete_cumsum
    dense_to_jagged
    jagged_to_padded_dense
    permute_pooled_embs
    permute2d_sparse_data
    split_embedding_codegen_forward_unweighted
    backward_codegen_adagrad_unweighted_exact
    dense_embedding_codegen_lookup_function
    dense_embedding_codegen_lookup_function_grad
)

if(NOT DEFINED FBGEMM_ASCEND_BUILD_VERS OR FBGEMM_ASCEND_BUILD_VERS STREQUAL "")
    set(FBGEMM_ASCEND_BUILD_VERS "A5,A2,A3")
endif()
string(REPLACE "," ";" FBGEMM_ASCEND_BUILD_VERS "${FBGEMM_ASCEND_BUILD_VERS}")

function(_fbgemm_get_target_info variant out_build out_ai)
    if(DEFINED ENV{FBGEMM_ASCEND_AI_CORE})
        if(variant STREQUAL "A5")
            set(${out_build} "c310" PARENT_SCOPE)
        else()
            set(${out_build} "v220" PARENT_SCOPE)
        endif()
        set(${out_ai} "$ENV{FBGEMM_ASCEND_AI_CORE}" PARENT_SCOPE)
        return()
    endif()
    if(variant STREQUAL "A5")
        set(${out_build} "c310" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend950" PARENT_SCOPE)
        return()
    elseif(variant STREQUAL "A2")
        set(${out_build} "v220" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend910B2" PARENT_SCOPE)
        return()
    elseif(variant STREQUAL "A3")
        set(${out_build} "v220" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend910_93" PARENT_SCOPE)
        return()
    endif()
    set(${out_build} "" PARENT_SCOPE)
    set(${out_ai} "" PARENT_SCOPE)
endfunction()

function(_fbgemm_add_ascendc_op vendor_name source_dir build_ver ai_core stage_dir variant)
    if(NOT ai_core)
        message(WARNING "ASCENDC ai_core is empty; skipping ${vendor_name} (${variant})")
        return()
    endif()

    set(work_dir "${source_dir}/${build_ver}")
    if(NOT EXISTS "${work_dir}/run.sh")
        message(STATUS "Skipping ${vendor_name} (${variant}); run.sh not found")
        return()
    endif()

    set(stamp "${CMAKE_CURRENT_BINARY_DIR}/${vendor_name}_${variant}.stamp")
    set(target_name "ascendc_${vendor_name}_${variant}")

    add_custom_command(
        OUTPUT ${stamp}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${stage_dir}
        COMMAND ${CMAKE_COMMAND} -E env
            FBGEMM_ASCEND_INSTALL_PATH=${stage_dir}
            bash ./run.sh ${ai_core}
        COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
        WORKING_DIRECTORY ${work_dir}
        DEPENDS ${work_dir}/run.sh
        COMMENT "Building AscendC ${vendor_name} (${variant})"
        VERBATIM)

    add_custom_target(${target_name} ALL DEPENDS ${stamp})
    get_property(_prev_target GLOBAL PROPERTY "FBGEMM_PREV_${vendor_name}" SET)
    if(_prev_target)
        get_property(_prev_name GLOBAL PROPERTY "FBGEMM_PREV_${vendor_name}")
        add_dependencies(${target_name} ${_prev_name})
    endif()
    set_property(GLOBAL PROPERTY "FBGEMM_PREV_${vendor_name}" "${target_name}")
    set_property(GLOBAL APPEND PROPERTY FBGEMM_ASCEND_TARGETS ${target_name})
endfunction()

set(_ASCENDC_OPS
    "float_or_half_to_fused_nbit_rowwise|${FBGEMM_ASCEND_SOURCE_DIR}/src/quantize_ops/float_or_half_to_fused_nbit_rowwise"
    "asynchronous_complete_cumsum|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/asynchronous_complete_cumsum"
    "block_bucketize_sparse_features|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/block_bucketize_sparse_features"
    "expand_into_jagged_permute|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/expand_into_jagged_permute"
    "invert_permute|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/invert_permute"
    "linearize_cache_indices|${FBGEMM_ASCEND_SOURCE_DIR}/src/split_embeddings_cache/linearize_cache_indices"
    "offsets_range|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/offsets_range"
    "permute2d_sparse_data|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/permute2d_sparse_data"
    "segment_sum_csr|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/segment_sum_csr"
    "init_address_lookup|${FBGEMM_ASCEND_SOURCE_DIR}/src/intraining_embedding_pruning_ops/init_address_lookup"
    "dense_to_jagged|${FBGEMM_ASCEND_SOURCE_DIR}/src/jagged_tensor_ops/dense_to_jagged"
    "jagged_to_padded_dense|${FBGEMM_ASCEND_SOURCE_DIR}/src/jagged_tensor_ops/jagged_to_padded_dense"
    "jagged_to_padded_dense_v2|${FBGEMM_ASCEND_SOURCE_DIR}/src/jagged_tensor_ops/jagged_to_padded_dense_v2"
    "select_dim1_to_permute|${FBGEMM_ASCEND_SOURCE_DIR}/src/jagged_tensor_ops/select_dim1_to_permute"
    "permute_pooled_embs|${FBGEMM_ASCEND_SOURCE_DIR}/src/pooled_embedding_ops/permute_pooled_embs"
    "int_nbit_split_embedding_codegen_lookup_function|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_inference/int_nbit_split_embedding_codegen_lookup_function"
    "pruned_hashmap_lookup|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_inference/pruned_hashmap_lookup"
    "backward_codegen_adagrad_unweighted_exact|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_training/backward_codegen_adagrad_unweighted_exact"
    "dense_embedding_codegen_lookup_function|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_training/dense_embedding_codegen_lookup_function"
    "dense_embedding_codegen_lookup_function_grad|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_training/dense_embedding_codegen_lookup_function_grad"
    "split_embedding_codegen_forward_unweighted|${FBGEMM_ASCEND_SOURCE_DIR}/src/tbe_training/split_embedding_codegen_forward_unweighted"
    "run_length_encode|${FBGEMM_ASCEND_SOURCE_DIR}/src/split_embeddings_cache/get_unique_indices"
    "lru_cache_find_uncached|${FBGEMM_ASCEND_SOURCE_DIR}/src/split_embeddings_cache/lru_cache_find_uncached"
    "lru_cache_insert_byte|${FBGEMM_ASCEND_SOURCE_DIR}/src/split_embeddings_cache/lru_cache_insert_byte"
    "group_index_select_dim0|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/group_index_select_dim0"
    "group_index_select_dim0_backward|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/group_index_select_dim0_backward"
)

foreach(_variant ${FBGEMM_ASCEND_BUILD_VERS})
    _fbgemm_get_target_info(${_variant} _build_ver _ascendc_ai_core)
    if(NOT _build_ver OR NOT _ascendc_ai_core)
        message(WARNING "Unknown variant ${_variant}; skipping")
        continue()
    endif()

    set(_stage_dir "${ASCENDC_STAGE_ROOT}/${_variant}")
    set(_vendors_for_config "")

    foreach(_op_entry ${_ASCENDC_OPS})
        string(REPLACE "|" ";" _parts "${_op_entry}")
        list(GET _parts 0 _vendor_name)
        list(GET _parts 1 _source_dir)

        if(_variant STREQUAL "A2")
            list(FIND ASCENDC_A5_ONLY_OPS ${_vendor_name} _a5_only_idx)
            if(_a5_only_idx GREATER -1)
                continue()
            endif()
        elseif(_variant STREQUAL "A3")
            list(FIND ASCENDC_A3_OPS ${_vendor_name} _a3_idx)
            if(_a3_idx EQUAL -1)
                continue()
            endif()
        endif()

        _fbgemm_add_ascendc_op(${_vendor_name} ${_source_dir} ${_build_ver} ${_ascendc_ai_core} ${_stage_dir} ${_variant})
        list(APPEND _vendors_for_config ${_vendor_name})
    endforeach()

    if(_vendors_for_config)
        string(REPLACE ";" "," _vendor_csv "${_vendors_for_config}")
        file(MAKE_DIRECTORY ${_stage_dir}/vendors)
        file(WRITE ${_stage_dir}/vendors/config.ini "load_priority=${_vendor_csv}\n")
        list(APPEND ASCENDC_STAGE_SUBDIRS ${_variant})
    endif()
endforeach()

list(REMOVE_DUPLICATES ASCENDC_STAGE_SUBDIRS)
get_property(ASCENDC_TARGETS GLOBAL PROPERTY FBGEMM_ASCEND_TARGETS)

# ---------------------------------------------------------------------------
# C++ 适配层源文件（新增算子在此追加）
# ---------------------------------------------------------------------------
set(FBGEMM_ASCEND_ADAPTER_SRCS
    src/quantize_ops/float_or_half_to_fused_nbit_rowwise/float_or_half_to_fused_nbit_rowwise.cpp
    src/sparse_ops/asynchronous_complete_cumsum/asynchronous_complete_cumsum.cpp
    src/sparse_ops/block_bucketize_sparse_features/block_bucketize_sparse_features.cpp
    src/sparse_ops/expand_into_jagged_permute/expand_into_jagged_permute.cpp
    src/sparse_ops/invert_permute/invert_permute.cpp
    src/split_embeddings_cache/linearize_cache_indices/linearize_cache_indices.cpp
    src/sparse_ops/offsets_range/offsets_range.cpp
    src/sparse_ops/permute2d_sparse_data/permute1d_sparse_data.cpp
    src/sparse_ops/permute2d_sparse_data/permute2d_sparse_data.cpp
    src/sparse_ops/segment_sum_csr/segment_sum_csr.cpp
    src/intraining_embedding_pruning_ops/init_address_lookup/init_address_lookup.cpp
    src/jagged_tensor_ops/dense_to_jagged/dense_to_jagged.cpp
    src/jagged_tensor_ops/jagged_to_padded_dense_v2/jagged_to_padded_dense_impl.cpp
    src/jagged_tensor_ops/jagged_to_padded_dense_v2/jagged_to_padded_dense_v1.cpp
    src/jagged_tensor_ops/jagged_to_padded_dense_v2/jagged_to_padded_dense_v2.cpp
    src/jagged_tensor_ops/select_dim1_to_permute/keyed_jagged_index_select_dim1.cpp
    src/pooled_embedding_ops/permute_pooled_embs/permute_pooled_embs.cpp
    src/tbe_inference/int_nbit_split_embedding_codegen_lookup_function/int_nbit_split_embedding_codegen_lookup_function.cpp
    src/tbe_inference/pruned_hashmap_lookup/pruned_hashmap_lookup.cpp
    src/tbe_training/dense_embedding_codegen_lookup_function/dense_embedding_codegen_lookup_function.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/split_embedding_codegen_forward_unweighted.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_adagrad_unweighted_exact.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_adagrad_unweighted_exact_grad_aggregation.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_adam_unweighted_exact.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_adam_unweighted_exact_grad_aggregation.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_rowwise_adagrad_unweighted_exact.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_sgd_unweighted_exact.cpp
    src/tbe_training/split_embedding_codegen_forward_unweighted/backward_codegen_sgd_unweighted_exact_grad_aggregation.cpp
    src/split_embeddings_cache/get_unique_indices/get_unique_indices.cpp
    src/split_embeddings_cache/lru_cache_populate_byte/lru_cache_populate_byte.cpp
    src/sparse_ops/group_index_select_dim0/group_index_select_dim0.cpp
)
