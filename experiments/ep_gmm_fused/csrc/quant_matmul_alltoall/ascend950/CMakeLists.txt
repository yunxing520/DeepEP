# Copyright (c) 2026 Huawei Technologies Co., Ltd.

set(QMMA_ASC_FLAGS
    "SHELL:--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false")

set(QMMA_CATCCOS_SOURCE_PATH "${QMMA_CATCCOS_SOURCE_PATH}")
set(QMMA_CATCCOS_TAG_ID "c845aa17559162c4d0f6801cc8dfd54d0ea2eb43")
if(NOT QMMA_CATCCOS_SOURCE_PATH AND DEFINED ENV{QMMA_CATCCOS_SOURCE_PATH})
    set(QMMA_CATCCOS_SOURCE_PATH "$ENV{QMMA_CATCCOS_SOURCE_PATH}")
endif()
if(NOT QMMA_CATCCOS_SOURCE_PATH)
    if(EXISTS "${CATCCOS_SOURCE_PATH}/include/catccos/dgemm/kernel/ascend950_quant_matmul_alltoall.hpp")
        set(QMMA_CATCCOS_SOURCE_PATH "${CATCCOS_SOURCE_PATH}")
    else()
        message(STATUS
            "Skipping ascend950_quant_matmul_alltoall: set QMMA_CATCCOS_SOURCE_PATH "
            "to the CatCCOS branch containing ascend950_quant_matmul_alltoall.hpp")
        return()
    endif()
endif()
get_filename_component(QMMA_CATCCOS_SOURCE_PATH "${QMMA_CATCCOS_SOURCE_PATH}" REALPATH)
if(NOT EXISTS "${QMMA_CATCCOS_SOURCE_PATH}/include/catccos/dgemm/kernel/ascend950_quant_matmul_alltoall.hpp")
    message(STATUS
        "Skipping ascend950_quant_matmul_alltoall: invalid QMMA_CATCCOS_SOURCE_PATH="
        "${QMMA_CATCCOS_SOURCE_PATH}")
    return()
endif()
if(NOT EXISTS "${QMMA_CATCCOS_SOURCE_PATH}/3rdparty/catlass/include/catlass/catlass.hpp")
    message(STATUS
        "Skipping ascend950_quant_matmul_alltoall: QMMA CatCCOS CATLASS submodule is missing")
    return()
endif()

# QMMA is currently developed on a CatCCOS feature branch while the shared
# ep_gmm_fused dependency remains pinned to its released revision. Validate the
# exact source revision here so a clean build cannot silently use another
# checkout or local edits.
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${QMMA_CATCCOS_SOURCE_PATH}"
    OUTPUT_VARIABLE QMMA_CATCCOS_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QMMA_CATCCOS_REV_STATUS)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD -- .
    WORKING_DIRECTORY "${QMMA_CATCCOS_SOURCE_PATH}"
    RESULT_VARIABLE QMMA_CATCCOS_DIRTY_STATUS)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff --cached --quiet -- .
    WORKING_DIRECTORY "${QMMA_CATCCOS_SOURCE_PATH}"
    RESULT_VARIABLE QMMA_CATCCOS_STAGED_STATUS)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD:3rdparty/catlass
    WORKING_DIRECTORY "${QMMA_CATCCOS_SOURCE_PATH}"
    OUTPUT_VARIABLE QMMA_CATLASS_EXPECTED_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QMMA_CATLASS_RESULT)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${QMMA_CATCCOS_SOURCE_PATH}/3rdparty/catlass"
    OUTPUT_VARIABLE QMMA_CATLASS_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QMMA_CATLASS_REV_STATUS)
if(NOT QMMA_CATCCOS_REV_STATUS EQUAL 0 OR NOT QMMA_CATCCOS_REV STREQUAL QMMA_CATCCOS_TAG_ID)
    message(FATAL_ERROR
        "QMMA CatCCOS revision mismatch: expected ${QMMA_CATCCOS_TAG_ID}, "
        "got '${QMMA_CATCCOS_REV}' from ${QMMA_CATCCOS_SOURCE_PATH}")
endif()
if(NOT QMMA_CATCCOS_DIRTY_STATUS EQUAL 0 OR NOT QMMA_CATCCOS_STAGED_STATUS EQUAL 0)
    message(FATAL_ERROR
        "QMMA CatCCOS has tracked or staged changes; commit them before building "
        "${QMMA_CATCCOS_SOURCE_PATH}")
endif()
if(NOT QMMA_CATLASS_RESULT EQUAL 0 OR NOT QMMA_CATLASS_REV_STATUS EQUAL 0 OR
   NOT QMMA_CATLASS_REV STREQUAL QMMA_CATLASS_EXPECTED_REV)
    message(FATAL_ERROR
        "QMMA CatCCOS CATLASS submodule is not initialized at its recorded revision")
endif()
message(STATUS
    "QMMA CatCCOS revision: ${QMMA_CATCCOS_REV}; CATLASS revision: ${QMMA_CATLASS_REV}")

set(QMMA_SHMEM_VERSION "")
set(QMMA_SHMEM_VERSION_FILE "")
foreach(VERSION_FILE IN ITEMS
        "${SHMEM_ROOT}/version.info"
        "${SHMEM_ROOT}/../version.info"
        "${SHMEM_SOURCE_PATH}/install/version.info")
    if(EXISTS "${VERSION_FILE}")
        file(READ "${VERSION_FILE}" QMMA_SHMEM_VERSION)
        set(QMMA_SHMEM_VERSION_FILE "${VERSION_FILE}")
        break()
    endif()
endforeach()
if(NOT QMMA_SHMEM_VERSION MATCHES "RDMA Backend : (XSCALE|HNS_1825)")
    message(STATUS
        "Skipping ascend950_quant_matmul_alltoall: SHMEM lacks RDMA backend XSCALE/HNS_1825")
    return()
endif()
set(QMMA_RDMA_BACKEND "${CMAKE_MATCH_1}")

set(QMMA_HNS_1825_PI_ON_CHIP "$ENV{SHMEM_HNS_1825_PI_ON_CHIP}" CACHE STRING
    "HNS_1825 doorbell mode: empty (SDK default), 0, or 1")
if(QMMA_HNS_1825_PI_ON_CHIP AND NOT QMMA_HNS_1825_PI_ON_CHIP MATCHES "^[01]$")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP must be empty, 0, or 1")
endif()
if(QMMA_RDMA_BACKEND STREQUAL "HNS_1825")
    if(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
        list(APPEND COMPILE_OPTIONS -DACLSHMEMI_HNS_1825_PI_ON_CHIP=${QMMA_HNS_1825_PI_ON_CHIP})
    endif()
elseif(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP is only valid for the HNS_1825 backend")
endif()

# Only this object sees the QMMA CatCCOS headers. Prepending keeps the branch
# ahead of the component-wide pinned CatCCOS include path.
set(INCLUDE_DIRECTORIES
    "${QMMA_CATCCOS_SOURCE_PATH}/include"
    "${QMMA_CATCCOS_SOURCE_PATH}/3rdparty/catlass/include"
    "${QMMA_CATCCOS_SOURCE_PATH}/tools"
    "${QMMA_CATCCOS_SOURCE_PATH}/utils"
    "${QMMA_CATCCOS_SOURCE_PATH}/examples/ascend950_quant_matmul_alltoall"
    ${INCLUDE_DIRECTORIES})
list(APPEND COMPILE_OPTIONS
    -DCATCCOS_SHMEM_WITH_QP=1
    -DCATCCOS_HIERARCHICAL_GROUP_SIZE=16
    -DASCEND_TIMER_A2A_MATMUL=1
    -DACLSHMEMI_RDMA_K_BACKEND_${QMMA_RDMA_BACKEND}=1)

set(QMMA_COMMON_OPTIONS
    ${COMPILE_OPTIONS}
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

set(QMMA_TORCH_NPU_PATH "${TORCH_NPU_PATH}")
if(NOT QMMA_TORCH_NPU_PATH)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch_npu; from pathlib import Path; print(Path(torch_npu.__file__).parent)"
        RESULT_VARIABLE QMMA_TORCH_NPU_QUERY_STATUS
        OUTPUT_VARIABLE QMMA_TORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT QMMA_TORCH_NPU_QUERY_STATUS EQUAL 0 OR NOT QMMA_TORCH_NPU_PATH)
        message(FATAL_ERROR "QMMA requires a Python environment with torch_npu")
    endif()
endif()
get_filename_component(QMMA_TORCH_NPU_PATH "${QMMA_TORCH_NPU_PATH}" REALPATH)
if(NOT EXISTS "${QMMA_TORCH_NPU_PATH}/include/torch_npu/csrc/core/npu/NPUStream.h")
    message(FATAL_ERROR "Invalid torch_npu include path: ${QMMA_TORCH_NPU_PATH}")
endif()
list(APPEND INCLUDE_DIRECTORIES
    "${QMMA_TORCH_NPU_PATH}/include"
    "${QMMA_TORCH_NPU_PATH}/..")

add_library(quant_matmul_alltoall_device_obj OBJECT quant_matmul_alltoall_device.cpp)
target_compile_options(quant_matmul_alltoall_device_obj PRIVATE
    -xasc
    -std=c++17
    ${QMMA_COMMON_OPTIONS}
    ${QMMA_ASC_FLAGS}
)
target_include_directories(quant_matmul_alltoall_device_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIRECTORIES}
)

add_library(quant_matmul_alltoall_host_obj OBJECT quant_matmul_alltoall_host.cpp)
target_compile_options(quant_matmul_alltoall_host_obj PRIVATE
    ${QMMA_COMMON_OPTIONS}
    -std=c++20
)
target_include_directories(quant_matmul_alltoall_host_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIRECTORIES}
)

set(QMMA_OBJECTS ${OBJECTS_LIST})
list(APPEND QMMA_OBJECTS
    $<TARGET_OBJECTS:quant_matmul_alltoall_device_obj>
    $<TARGET_OBJECTS:quant_matmul_alltoall_host_obj>)
set(OBJECTS_LIST ${QMMA_OBJECTS} CACHE INTERNAL "CatCCOS UDMA operator objects" FORCE)
