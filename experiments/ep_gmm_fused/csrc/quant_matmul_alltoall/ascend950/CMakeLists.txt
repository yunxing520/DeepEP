# Copyright (c) 2026 Huawei Technologies Co., Ltd.

set(CATCCOS_HIERARCHICAL_GROUP_SIZE 16 CACHE STRING "Consecutive ranks per UDMA domain")
if(NOT CATCCOS_HIERARCHICAL_GROUP_SIZE MATCHES "^[1-9][0-9]*$"
   OR CATCCOS_HIERARCHICAL_GROUP_SIZE GREATER 32)
    message(FATAL_ERROR "CATCCOS_HIERARCHICAL_GROUP_SIZE must be in [1, 32]")
endif()

# Dependencies.cmake validates the HNS_1825 SDK and records its compiled
# doorbell mode. Use the same settings as the core RoCE kernels.
list(APPEND COMPILE_OPTIONS
    -DCATCCOS_SHMEM_WITH_QP=1
    -DCATCCOS_HIERARCHICAL_GROUP_SIZE=${CATCCOS_HIERARCHICAL_GROUP_SIZE}
    -DASCEND_TIMER_A2A_MATMUL=1
    -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1
    -DACLSHMEMI_HNS_1825_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP})

set(QUANT_MATMUL_ALL_TO_ALL_ASC_FLAGS
    "--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_sources("${QUANT_MATMUL_ALL_TO_ALL_ASC_FLAGS}")
