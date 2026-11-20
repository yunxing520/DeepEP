# Copyright (c) 2026 Huawei Technologies Co., Ltd.

set(CATCCOS_HIERARCHICAL_GROUP_SIZE 16 CACHE STRING "Consecutive ranks per UDMA domain")
if(NOT CATCCOS_HIERARCHICAL_GROUP_SIZE MATCHES "^[1-9][0-9]*$"
   OR CATCCOS_HIERARCHICAL_GROUP_SIZE GREATER 32)
    message(FATAL_ERROR "CATCCOS_HIERARCHICAL_GROUP_SIZE must be in [1, 32]")
endif()

# SHMEM's installed headers do not export the selected RDMA backend. The QMMA
# multi-QP device path must compile against the backend recorded by its SDK.
set(QMMA_SHMEM_VERSION "")
foreach(version_file IN ITEMS
        "${SHMEM_ROOT}/version.info"
        "${SHMEM_ROOT}/../version.info"
        "${SHMEM_SOURCE_PATH}/install/version.info")
    if(EXISTS "${version_file}")
        file(READ "${version_file}" QMMA_SHMEM_VERSION)
        break()
    endif()
endforeach()
if(NOT QMMA_SHMEM_VERSION MATCHES "RDMA Backend : (XSCALE|HNS_1825)")
    message(FATAL_ERROR "QMMA requires a SHMEM SDK built with the XSCALE or HNS_1825 RDMA backend")
endif()
set(QMMA_RDMA_BACKEND "${CMAKE_MATCH_1}")

set(QMMA_HNS_1825_PI_ON_CHIP "$ENV{SHMEM_HNS_1825_PI_ON_CHIP}" CACHE STRING
    "HNS_1825 doorbell mode: empty (SDK default), 0, or 1")
if(QMMA_HNS_1825_PI_ON_CHIP AND NOT QMMA_HNS_1825_PI_ON_CHIP MATCHES "^[01]$")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP must be empty, 0, or 1")
endif()

list(APPEND COMPILE_OPTIONS
    -DCATCCOS_SHMEM_WITH_QP=1
    -DCATCCOS_HIERARCHICAL_GROUP_SIZE=${CATCCOS_HIERARCHICAL_GROUP_SIZE}
    -DASCEND_TIMER_A2A_MATMUL=1
    -DACLSHMEMI_RDMA_K_BACKEND_${QMMA_RDMA_BACKEND}=1)
if(QMMA_RDMA_BACKEND STREQUAL "HNS_1825")
    if(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
        list(APPEND COMPILE_OPTIONS -DACLSHMEMI_HNS_1825_PI_ON_CHIP=${QMMA_HNS_1825_PI_ON_CHIP})
    endif()
elseif(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP is only valid for the HNS_1825 backend")
endif()

set(QUANT_MATMUL_ALL_TO_ALL_ASC_FLAGS
    "--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_sources("${QUANT_MATMUL_ALL_TO_ALL_ASC_FLAGS}")
