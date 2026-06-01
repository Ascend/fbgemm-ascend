# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

set(FBGEMM_ASCEND_CODEGEN_DIR ${FBGEMM_ASCEND_SOURCE_DIR}/codegen)
set(FBGEMM_ASCEND_TBE_INFERENCE_WEIGHT_TYPES "FP8,FP16,FP32,INT8,INT4,INT2" CACHE STRING
    "Comma-separated TBE inference weight types to generate for fbgemm-ascend")

set(FBGEMM_ASCEND_TBE_INFERENCE_CODEGEN_SCRIPT
    ${FBGEMM_ASCEND_CODEGEN_DIR}/genscript/generate_tbe_inference.py)
set(FBGEMM_ASCEND_TBE_INFERENCE_GENERATED_ROOT
    ${CMAKE_CURRENT_BINARY_DIR}/generated/tbe_inference)
set(FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT
    ${FBGEMM_ASCEND_TBE_INFERENCE_GENERATED_ROOT}/int_nbit_split_embedding_codegen_lookup_function)

set(FBGEMM_ASCEND_TBE_INFERENCE_CODEGEN_DEPS
    ${FBGEMM_ASCEND_CODEGEN_DIR}/genscript/common.py
    ${FBGEMM_ASCEND_CODEGEN_DIR}/genscript/generate_int_nbit_split_embedding_codegen_lookup_function.py
    ${FBGEMM_ASCEND_CODEGEN_DIR}/genscript/generate_tbe_inference.py
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_adapter_template.cpp
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_common_template.h
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_host_op_template.cpp
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_host_tiling_template.h
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_json_template.json
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_kernel_entry_template.cpp
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_kernel_nobag_template.h
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_kernel_pooling_template.h
    ${FBGEMM_ASCEND_CODEGEN_DIR}/inference/nbit_run_template.sh
)

function(fbgemm_ascend_run_tbe_inference_codegen)
    # Create output directory if it doesn't exist
    file(MAKE_DIRECTORY ${FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT})

    add_custom_target(fbgemm_ascend_tbe_inference_codegen
        COMMAND ${Python3_EXECUTABLE}
            ${FBGEMM_ASCEND_TBE_INFERENCE_CODEGEN_SCRIPT}
            --install-dir ${FBGEMM_ASCEND_SOURCE_DIR}
            --output-dir ${FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT}
            --weight-types ${FBGEMM_ASCEND_TBE_INFERENCE_WEIGHT_TYPES}
        WORKING_DIRECTORY ${FBGEMM_ASCEND_SOURCE_DIR}
        DEPENDS ${FBGEMM_ASCEND_TBE_INFERENCE_CODEGEN_DEPS}
        COMMENT "Generating fbgemm-ascend TBE inference sources"
        VERBATIM)

    execute_process(
        COMMAND ${Python3_EXECUTABLE}
            ${FBGEMM_ASCEND_TBE_INFERENCE_CODEGEN_SCRIPT}
            --install-dir ${FBGEMM_ASCEND_SOURCE_DIR}
            --output-dir ${FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT}
            --weight-types ${FBGEMM_ASCEND_TBE_INFERENCE_WEIGHT_TYPES}
        WORKING_DIRECTORY ${FBGEMM_ASCEND_SOURCE_DIR}
        RESULT_VARIABLE _codegen_result
    )

    if(NOT _codegen_result EQUAL 0)
        message(FATAL_ERROR "fbgemm-ascend TBE inference code generation failed")
    endif()

    set(FBGEMM_ASCEND_TBE_INFERENCE_ASCENDC_OPS
        "int_nbit_split_embedding_codegen_lookup_function|${FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT}"
        PARENT_SCOPE)
    set(FBGEMM_ASCEND_TBE_INFERENCE_ADAPTER_SRCS_A5_ONLY
        "${FBGEMM_ASCEND_TBE_INFERENCE_LOOKUP_FUNCTION_ROOT}/int_nbit_split_embedding_codegen_lookup_function.cpp"
        PARENT_SCOPE)
    set(FBGEMM_ASCEND_TBE_INFERENCE_A5_ONLY_OPS
        int_nbit_split_embedding_codegen_lookup_function
        PARENT_SCOPE)
endfunction()
