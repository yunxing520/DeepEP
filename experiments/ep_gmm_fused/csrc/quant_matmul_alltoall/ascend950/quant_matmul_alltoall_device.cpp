/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "catccos/comm/kernel/ascend950_hierarchical_barrier.hpp"
#include "ascend950_quant_matmul_alltoall_device.h"
#include "quant_matmul_alltoall_bridge.h"

#include <acl/acl.h>

#include <cstdint>

namespace {

using QmmaBarrier = Catccos::Comm::Kernel::HierarchicalBarrier;
using QmmaLayout = Catccos::DGemm::Kernel::detail::QuantA2ASymLayout;

uint32_t Rounds(const QmmaLaunchArgs& args)
{
    return Catccos::DGemm::Kernel::detail::QuantA2ARoundCount(args.m, args.comm_interval * 128U);
}

uint32_t Q(const QmmaLaunchArgs& args)
{
    return args.n / args.world_size;
}

uint64_t TxBytes(const QmmaLaunchArgs& args)
{
    return QmmaLayout::TxBytes(Rounds(args), args.world_size, args.comm_interval * 128U, Q(args), sizeof(uint16_t));
}

uint64_t YBytes(const QmmaLaunchArgs& args)
{
    return QmmaLayout::YBytes(args.m, args.world_size, Q(args), sizeof(uint16_t));
}

uint64_t ControlOffset(const QmmaLaunchArgs& args)
{
    return TxBytes(args) + YBytes(args);
}

template <class Config>
int32_t Launch(const QmmaLaunchArgs& args)
{
    using DeviceOp = typename Config::Device;
    typename DeviceOp::Arguments kernelArgs{
        Catlass::GemmCoord{args.m, args.n, args.k},
        args.rank,
        args.world_size,
        args.ranks_per_node,
        args.aic_num,
        args.comm_interval,
        args.lane_num,
        args.epoch,
        args.a,
        args.b,
        args.a_scale,
        args.b_scale,
        args.out,
        args.symmetric,
    };
    DeviceOp operation;
    if (operation.Initialize(kernelArgs) != Catlass::Status::kSuccess) {
        return 1;
    }
    return operation.Run(static_cast<aclrtStream>(args.stream), args.aic_num, args.ffts_address) ==
                   Catlass::Status::kSuccess
               ? 0
               : 2;
}

}  // namespace

extern "C" uint64_t qmma_symmetric_bytes(const QmmaLaunchArgs* args)
{
    const uint32_t h = args->comm_interval * 128U;
    const uint32_t q = args->n / args->world_size;
    const uint32_t rounds = Catccos::DGemm::Kernel::detail::QuantA2ARoundCount(args->m, h);
    return QmmaLayout::TotalBytes(rounds, args->world_size, args->m, h, q, sizeof(uint16_t), args->aic_num);
}

extern "C" uint64_t qmma_output_offset(const QmmaLaunchArgs* args)
{
    return TxBytes(*args);
}

extern "C" uint64_t qmma_control_offset(const QmmaLaunchArgs* args)
{
    return ControlOffset(*args);
}

extern "C" uint64_t qmma_control_bytes(const QmmaLaunchArgs* args)
{
    return qmma_symmetric_bytes(args) - ControlOffset(*args);
}

extern "C" int32_t qmma_launch(const QmmaLaunchArgs* args)
{
    if (args->format == 0) {
        using Config = Ascend950QuantMatmulAllToAllConfigFp8<
            float8_e4m3_t, Catlass::layout::RowMajor, float8_e4m3_t, Catlass::layout::RowMajor, float8_e8m0_t,
            Catlass::layout::RowMajor, float8_e8m0_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor>;
        return Launch<Config>(*args);
    }
    if (args->format == 1) {
        using Config = Ascend950QuantMatmulAllToAllConfigFp4<
            float4_e2m1x2_t, Catlass::layout::RowMajor, float4_e2m1x2_t, Catlass::layout::RowMajor, float8_e8m0_t,
            Catlass::layout::RowMajor, float8_e8m0_t, Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor>;
        return Launch<Config>(*args);
    }
    using Config =
        Ascend950QuantMatmulAllToAllConfigBf16<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                               Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor>;
    return Launch<Config>(*args);
}

extern "C" uint64_t qmma_barrier_bytes(uint32_t world_size, uint32_t ranks_per_node)
{
    const QmmaBarrier::Arguments args{0U, world_size, ranks_per_node, 1U};
    return QmmaBarrier::GetWorkspaceSize(args);
}

extern "C" int32_t qmma_hierarchical_barrier(uint32_t rank, uint32_t world_size, uint32_t ranks_per_node,
                                             uint32_t epoch, uint8_t* workspace, void* stream, uint64_t ffts_address)
{
    const QmmaBarrier::Arguments args{rank, world_size, ranks_per_node, epoch};
    const auto params = QmmaBarrier::ToUnderlyingArguments(args, workspace);
    if (ffts_address == 0) {
        Catccos::DGemm::Device::KernelAdapter<QmmaBarrier>
            <<<1, nullptr, static_cast<aclrtStream>(stream)>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<QmmaBarrier>
            <<<1, nullptr, static_cast<aclrtStream>(stream)>>>(params, ffts_address, nullptr);
    }
    return aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) == ACL_SUCCESS ? 0 : 3;
}
