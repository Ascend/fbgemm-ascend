# fbgemm_ascend: 算子适配层构建
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fbgemm_ascend LANGUAGES CXX VERSION 0.1.0)

set(FBGEMM_ASCEND_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Show compile commands" FORCE)

# -----------------------------------------------------------------------------
# Python / PyTorch / torch_npu 路径（优先使用 setup.py 传入的 CMAKE_PREFIX_PATH）
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(CMAKE_PREFIX_PATH)
    set(PYTORCH_INSTALL_PATH "${CMAKE_PREFIX_PATH}")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTORCH_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch")
endif()
get_filename_component(PYTHON_SITE_PACKAGES "${PYTORCH_INSTALL_PATH}/.." ABSOLUTE)
set(PYTORCH_NPU_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch_npu")
if(NOT EXISTS "${PYTORCH_NPU_INSTALL_PATH}")
    message(WARNING "torch_npu not found at ${PYTORCH_NPU_INSTALL_PATH}; NPU build may fail.")
endif()

# CANN 系统路径：securec.h 等在 CANN 包路径下 ${arch}-linux/include
# 优先使用系统默认的 ASCEND_HOME_PATH，其次允许自定义 ASCEND_CANN_PACKAGE_PATH
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
else()
    set(ASCEND_CANN_PACKAGE_PATH "")
endif()
if(ASCEND_CANN_PACKAGE_PATH)
    set(CANN_INCLUDE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    if(IS_DIRECTORY "${CANN_INCLUDE_DIR}")
        message(STATUS "CANN include (securec.h etc.): ${CANN_INCLUDE_DIR}")
    else()
        message(WARNING "CANN include dir not found: ${CANN_INCLUDE_DIR}; set ASCEND_CANN_PACKAGE_PATH to your CANN root (e.g. /usr/local/Ascend/cann-9.0.T501).")
        set(CANN_INCLUDE_DIR "")
    endif()
else()
    set(CANN_INCLUDE_DIR "")
    message(STATUS "ASCEND_CANN_PACKAGE_PATH not set; if securec.h is missing, set it to CANN root (e.g. /usr/local/Ascend/cann-9.0.T501).")
endif()

# CANN driver：部分环境在 driver 下提供 libc_sec/include（与 CommonTorchOpConfig.cmake 一致，作备选）
set(ASCEND_DRIVER_PATH "$ENV{ASCEND_DRIVER_PATH}")
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()
if(IS_DIRECTORY "${ASCEND_DRIVER_PATH}/kernel/libc_sec/include")
    set(DRIVER_LIBC_SEC_INCLUDE "${ASCEND_DRIVER_PATH}/kernel/libc_sec/include")
else()
    set(DRIVER_LIBC_SEC_INCLUDE "")
endif()

# ABI：必须与当前 PyTorch 一致（1=C++11 std::string 新 ABI，0=旧 ABI）；符号不一致会导致 load_library 报 undefined symbol
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; v=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 0); print(1 if v else 0)"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if("${TORCH_CXX11_ABI}" STREQUAL "" OR "${TORCH_CXX11_ABI}" STREQUAL "True" OR "${TORCH_CXX11_ABI}" STREQUAL "False")
    set(TORCH_CXX11_ABI 0)
endif()
if(NOT "${TORCH_CXX11_ABI}" STREQUAL "1")
    set(TORCH_CXX11_ABI 0)
endif()
message(STATUS "TORCH_CXX11_ABI=${TORCH_CXX11_ABI} (must match torch lib for torch::Library symbol)")

# -----------------------------------------------------------------------------
# AscendC 算子编译与打包（逻辑在 FbgemmAscend.cmake，新增算子只需改 FbgemmAscend.cmake 文件）
# -----------------------------------------------------------------------------
include(${FBGEMM_ASCEND_SOURCE_DIR}/FbgemmAscend.cmake)

# -----------------------------------------------------------------------------
# 主库：针对 A5 (c310) 与 A2/A3 (v220) 构建两份适配层
# -----------------------------------------------------------------------------

function(fbgemm_add_host_library target_name output_name npu_chip_flag)
    add_library(${target_name} SHARED ${FBGEMM_ASCEND_ADAPTER_SRCS})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
        -DNPU_CHIP_A5=${npu_chip_flag}
    )

    set(FBGEMM_ASCEND_EXTRA_INCLUDES "")
    if(CANN_INCLUDE_DIR)
        list(APPEND FBGEMM_ASCEND_EXTRA_INCLUDES ${CANN_INCLUDE_DIR})
    endif()
    if(DRIVER_LIBC_SEC_INCLUDE)
        list(APPEND FBGEMM_ASCEND_EXTRA_INCLUDES ${DRIVER_LIBC_SEC_INCLUDE})
    endif()
    target_include_directories(${target_name} PRIVATE
        ${FBGEMM_ASCEND_SOURCE_DIR}/src
        ${FBGEMM_ASCEND_SOURCE_DIR}/include
        ${PYTORCH_NPU_INSTALL_PATH}/include
        ${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc
        ${PYTORCH_INSTALL_PATH}/include
        ${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
        ${Python3_INCLUDE_DIRS}
        ${FBGEMM_ASCEND_EXTRA_INCLUDES}
    )

    target_link_directories(${target_name} PRIVATE
        ${PYTORCH_INSTALL_PATH}/lib
        ${PYTORCH_NPU_INSTALL_PATH}/lib
    )

    set(FBGEMM_ASCEND_TORCH_LIBS c10 torch torch_cpu torch_npu)
    find_library(TORCH_PYTHON_LIB torch_python HINTS "${PYTORCH_INSTALL_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIB)
        list(APPEND FBGEMM_ASCEND_TORCH_LIBS torch_python)
        message(STATUS "Linking ${target_name} with torch_python: ${TORCH_PYTHON_LIB}")
    else()
        message(STATUS "torch_python not found for ${target_name}; if load_library shows undefined symbol torch::Library, rebuild PyTorch or use a build that exports it.")
    endif()
    target_link_libraries(${target_name} PRIVATE ${FBGEMM_ASCEND_TORCH_LIBS} Python3::Python)

    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${output_name}
        BUILD_RPATH "${PYTORCH_INSTALL_PATH}/lib;${PYTORCH_NPU_INSTALL_PATH}/lib"
        INSTALL_RPATH "${PYTORCH_INSTALL_PATH}/lib;${PYTORCH_NPU_INSTALL_PATH}/lib"
    )
endfunction()

set(BUILD_HOST_A5 OFF)
set(BUILD_HOST_A2A3 OFF)
foreach(_variant ${FBGEMM_ASCEND_BUILD_VERS})
    if(_variant STREQUAL "A5")
        set(BUILD_HOST_A5 ON)
    elseif(_variant STREQUAL "A2" OR _variant STREQUAL "A3")
        set(BUILD_HOST_A2A3 ON)
    endif()
endforeach()
if(NOT BUILD_HOST_A5 AND NOT BUILD_HOST_A2A3)
    set(BUILD_HOST_A5 ON)
endif()

if(BUILD_HOST_A5)
    fbgemm_add_host_library(fbgemm_ascend_py_a5 fbgemm_ascend_py_a5 1)
endif()
if(BUILD_HOST_A2A3)
    fbgemm_add_host_library(fbgemm_ascend_py_a2a3 fbgemm_ascend_py_a2a3 0)
endif()

foreach(_t ${ASCENDC_TARGETS})
    if(TARGET ${_t})
        if(TARGET fbgemm_ascend_py_a5)
            add_dependencies(fbgemm_ascend_py_a5 ${_t})
        endif()
        if(TARGET fbgemm_ascend_py_a2a3)
            add_dependencies(fbgemm_ascend_py_a2a3 ${_t})
        endif()
    endif()
endforeach()

# 若环境中存在 Ascend driver 的 c_sec，可取消注释并链接
# target_link_directories(fbgemm_ascend_py_a5 PRIVATE /usr/local/Ascend/driver/lib64/common)
# target_link_libraries(fbgemm_ascend_py_a5 PRIVATE c_sec)
# target_link_directories(fbgemm_ascend_py_a2a3 PRIVATE /usr/local/Ascend/driver/lib64/common)
# target_link_libraries(fbgemm_ascend_py_a2a3 PRIVATE c_sec)

# -----------------------------------------------------------------------------
# 安装：将 .so 安装到 Python 包目录 fbgemm_ascend/
# -----------------------------------------------------------------------------
if(TARGET fbgemm_ascend_py_a5)
    install(TARGETS fbgemm_ascend_py_a5
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()
if(TARGET fbgemm_ascend_py_a2a3)
    install(TARGETS fbgemm_ascend_py_a2a3
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()

# 将 staging 目录中的 AscendC 算子文件打入 whl 包
if(ASCENDC_STAGE_SUBDIRS)
    foreach(_opp_ver ${ASCENDC_STAGE_SUBDIRS})
        install(DIRECTORY ${ASCENDC_STAGE_ROOT}/${_opp_ver}/
            DESTINATION opp/${_opp_ver}
            OPTIONAL)
    endforeach()
endif()

# 将 env_setup.sh 打入包目录，供用户 source
install(FILES ${FBGEMM_ASCEND_SOURCE_DIR}/fbgemm_ascend/env_setup.sh
    DESTINATION .
)
