/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef DEEPEP_EP_GMM_FUSED_QUANT_MATMUL_ALLTOALL_BRIDGE_H
#define DEEPEP_EP_GMM_FUSED_QUANT_MATMUL_ALLTOALL_BRIDGE_H

#include <cstdint>

// ABI shared by the Torch host wrapper and the CatCCOS device launcher.
// Keep ATen, SHMEM and AscendC types out of this header.
struct QmmaLaunchArgs {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rank;
    uint32_t world_size;
    uint32_t ranks_per_node;
    uint32_t aic_num;
    uint32_t comm_interval;
    uint32_t lane_num;
    uint32_t epoch;
    uint32_t format;
    uint8_t* a;
    uint8_t* b;
    uint8_t* a_scale;
    uint8_t* b_scale;
    uint8_t* out;
    uint8_t* symmetric;
    void* stream;
    uint64_t ffts_address;
};

extern "C" uint64_t qmma_symmetric_bytes(const QmmaLaunchArgs* args);
extern "C" uint64_t qmma_output_offset(const QmmaLaunchArgs* args);
extern "C" uint64_t qmma_control_offset(const QmmaLaunchArgs* args);
extern "C" uint64_t qmma_control_bytes(const QmmaLaunchArgs* args);
extern "C" int32_t qmma_launch(const QmmaLaunchArgs* args);
extern "C" int32_t qmma_hierarchical_barrier(uint32_t rank, uint32_t world_size, uint32_t ranks_per_node,
                                             uint32_t epoch, uint8_t* workspace, void* stream, uint64_t ffts_address);
extern "C" uint64_t qmma_barrier_bytes(uint32_t world_size, uint32_t ranks_per_node);

#endif  // DEEPEP_EP_GMM_FUSED_QUANT_MATMUL_ALLTOALL_BRIDGE_H
