/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "../../tensor_validation.hpp"
#include "catccos/comm/kernel/ascend950_hierarchical_barrier.hpp"
#include "catccos/dgemm/kernel/ascend950_quant_matmul_alltoall.hpp"
#include "quant_matmul_alltoall_device.hpp"
#include <acl/acl.h>
#include "platform/platform_ascendc.h"
#include "shmem.h"
#include "shmem_init.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#ifndef EXTENSION_MODULE_NAME
#error "EXTENSION_MODULE_NAME must be provided by the EP_GMM_FUSED build"
#endif

using namespace AscendC;

namespace deepep::ep_gmm_fused {
namespace {
struct QmmaLaunchArgs {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rank;
    uint32_t world_size;
    uint32_t ranks_per_node;
    uint32_t aic_num;
    uint32_t comm_interval;
    uint32_t tile_m;
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

using QmmaBarrier = Catccos::Comm::Kernel::HierarchicalBarrier;
using QmmaLayout = Catccos::DGemm::Kernel::detail::QuantA2ASymLayout;

uint32_t TileM(const QmmaLaunchArgs& args)
{
    return args.tile_m;
}

uint32_t Rounds(const QmmaLaunchArgs& args)
{
    return Catccos::DGemm::Kernel::detail::QuantA2ARoundCount(args.m, args.comm_interval * TileM(args));
}

uint32_t Q(const QmmaLaunchArgs& args)
{
    return args.n / args.world_size;
}

/// TX 采用双缓冲:kernel 只使用 min(rounds, TX_STAGES=2) 个物理槽。
uint32_t Slots(const QmmaLaunchArgs& args)
{
    const uint32_t rounds = Rounds(args);
    return rounds < 2U ? rounds : 2U;
}

uint64_t TxBytes(const QmmaLaunchArgs& args)
{
    return QmmaLayout::TxBytes(Slots(args), args.world_size, args.comm_interval * TileM(args), Q(args),
                               sizeof(uint16_t));
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

uint64_t SymmetricBytes(const QmmaLaunchArgs* args)
{
    const uint32_t h = args->comm_interval * TileM(*args);
    const uint32_t q = args->n / args->world_size;
    const uint32_t rounds = Catccos::DGemm::Kernel::detail::QuantA2ARoundCount(args->m, h);
    const uint32_t slots = rounds < 2U ? rounds : 2U;
    return QmmaLayout::TotalBytes(slots, args->world_size, args->m, h, q, sizeof(uint16_t), args->aic_num);
}

uint64_t OutputOffset(const QmmaLaunchArgs* args)
{
    return TxBytes(*args);
}

uint64_t ControlRegionOffset(const QmmaLaunchArgs* args)
{
    return ControlOffset(*args);
}

uint64_t ControlRegionBytes(const QmmaLaunchArgs* args)
{
    return SymmetricBytes(args) - ControlOffset(*args);
}

int32_t LaunchQuantMatmulAllToAll(const QmmaLaunchArgs* args)
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
    if ((args->n / args->world_size) % 256 != 0) {
        using Config = Ascend950QuantMatmulAllToAllConfigBf16N128<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                                  Catlass::layout::RowMajor, bfloat16_t,
                                                                  Catlass::layout::RowMajor>;
        return Launch<Config>(*args);
    }
    using Config =
        Ascend950QuantMatmulAllToAllConfigBf16<bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                               Catlass::layout::RowMajor, bfloat16_t, Catlass::layout::RowMajor>;
    return Launch<Config>(*args);
}

uint64_t BarrierWorkspaceBytes(uint32_t world_size, uint32_t ranks_per_node)
{
    const QmmaBarrier::Arguments args{0U, world_size, ranks_per_node, 1U};
    return QmmaBarrier::GetWorkspaceSize(args);
}

int32_t LaunchHierarchicalBarrier(uint32_t rank, uint32_t world_size, uint32_t ranks_per_node, uint32_t epoch,
                                  uint8_t* workspace, void* stream, uint64_t ffts_address)
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

constexpr uint32_t kRanksPerNode = CATCCOS_HIERARCHICAL_GROUP_SIZE;
constexpr uint32_t kMaxWorldSize = std::min(32U, 2U * kRanksPerNode);
constexpr uint64_t kMiB = 1024ULL * 1024ULL;
constexpr uint64_t kHeapAlignment = 2ULL * kMiB;
constexpr uint64_t kHeapReserve = 16ULL * kMiB;
constexpr uint32_t kDefaultTimeout = 60;

struct ProblemShape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

/// kernel 内 h = commInterval * L1_TILE_M(编译期 M tile),host 侧必须用同一基准,
/// 否则 rounds/TX/prodTicket 申请不足(kernel 侧 h 为 8192 而 host 按 4096 计)会越界(aclError 507015)。
uint32_t TileMFor(const ProblemShape& shape, int64_t format, int64_t epSize)
{
    if (format == 2 && epSize > 0 && (shape.n / epSize) % 256U != 0U) {
        return 256U;
    }
    return 128U;
}

QmmaLaunchArgs MakeLaunchArgs(const ProblemShape& shape, int64_t format, int64_t epSize, int64_t commInterval,
                              uint32_t rank, uint32_t lanes, uint32_t aicNum, uint32_t epoch, uint8_t* symmetric,
                              const void* a, const void* b, const void* aScale, const void* bScale, void* output,
                              aclrtStream stream, uint64_t fftsAddress)
{
    return QmmaLaunchArgs{
        shape.m,
        shape.n,
        shape.k,
        rank,
        static_cast<uint32_t>(epSize),
        kRanksPerNode,
        aicNum,
        static_cast<uint32_t>(commInterval),
        TileMFor(shape, format, epSize),
        lanes,
        epoch,
        static_cast<uint32_t>(format),
        const_cast<uint8_t*>(static_cast<const uint8_t*>(a)),
        const_cast<uint8_t*>(static_cast<const uint8_t*>(b)),
        const_cast<uint8_t*>(static_cast<const uint8_t*>(aScale)),
        const_cast<uint8_t*>(static_cast<const uint8_t*>(bScale)),
        static_cast<uint8_t*>(output),
        symmetric,
        stream,
        fftsAddress,
    };
}

QmmaLaunchArgs MakeLaunchArgs(const ProblemShape& shape, int64_t format, int64_t epSize, int64_t commInterval,
                              uint32_t rank, uint32_t lanes, uint32_t aicNum, uint32_t epoch)
{
    return MakeLaunchArgs(shape, format, epSize, commInterval, rank, lanes, aicNum, epoch, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, 0);
}

uint64_t SelectHeapBytes(uint64_t required)
{
    TORCH_CHECK(required <= std::numeric_limits<uint64_t>::max() - kHeapReserve,
                "QMMA symmetric workspace overflows heap sizing");
    const uint64_t rounded = AlignUp(required + kHeapReserve, kHeapAlignment);
    const uint64_t maxUniqueIdHeap = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    TORCH_CHECK(rounded <= maxUniqueIdHeap, "QMMA symmetric workspace exceeds the SHMEM heap range");
    return rounded;
}

class QmmaRuntime {
public:
    static QmmaRuntime& Instance()
    {
        static QmmaRuntime runtime;
        return runtime;
    }

    static int64_t UniqueIdSize()
    {
        return static_cast<int64_t>(sizeof(aclshmemx_uniqueid_t));
    }

    static std::vector<int64_t> UniqueId()
    {
        aclshmemx_uniqueid_t uid{};
        TORCH_CHECK(aclshmemx_get_uniqueid(&uid) == ACLSHMEM_SUCCESS, "failed to create the QMMA SHMEM unique ID");
        std::vector<int64_t> values(sizeof(uid));
        const auto* bytes = reinterpret_cast<const uint8_t*>(&uid);
        for (size_t i = 0; i < sizeof(uid); ++i) {
            values[i] = bytes[i];
        }
        return values;
    }

    void Initialize(int32_t rank, int32_t worldSize, uint32_t lanes, uint64_t minimumHeapSize,
                    const std::string& endpoint, const std::vector<uint8_t>& uid)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        TORCH_CHECK(rank >= 0 && rank < worldSize && worldSize >= 2 && worldSize <= kMaxWorldSize,
                    "invalid QMMA topology");
        TORCH_CHECK(lanes >= 1 && lanes <= 4, "QMMA lanes must be in [1, 4]");
        TORCH_CHECK(minimumHeapSize > 0, "QMMA heap_size must be positive");
        TORCH_CHECK(endpoint.size() < ACLSHMEM_MAX_IP_PORT_LEN, "QMMA bootstrap endpoint is too long");
        TORCH_CHECK(endpoint.find('\0') == std::string::npos, "QMMA bootstrap endpoint contains NUL");
        TORCH_CHECK(uid.empty() || uid.size() == UniqueIdSize(), "invalid QMMA SHMEM unique ID");
        TORCH_CHECK(!uid.empty() || !endpoint.empty(), "QMMA bootstrap endpoint is empty");
        TORCH_CHECK(!initialized_ && !initializing_, "QMMA runtime is already initialized");
        TORCH_CHECK(aclshmemx_init_status() == ACLSHMEM_STATUS_NOT_INITIALIZED,
                    "another SHMEM runtime exists in this process");

        int32_t device = -1;
        TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "cannot query the active NPU");
        const char* soc = aclrtGetSocName();
        TORCH_CHECK(soc != nullptr && std::string(soc).rfind("Ascend950", 0) == 0,
                    "QMMA is supported only on Ascend950");

        const uint32_t aicNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        TORCH_CHECK(aicNum > 0 && aicNum <= 64 && 2U * aicNum >= static_cast<uint32_t>(worldSize),
                    "insufficient AIC resources for QMMA");

        rank_ = rank;
        worldSize_ = worldSize;
        device_ = device;
        lanes_ = lanes;
        aicNum_ = aicNum;
        heapBytes_ = SelectHeapBytes(minimumHeapSize);
        endpoint_ = endpoint;
        uid_ = uid;
        initializing_ = true;
        try {
            InitializeShmem();
            initialized_ = true;
            initializing_ = false;
        } catch (...) {
            PoisonShmem();
            throw;
        }
    }

    void Destroy(aclrtStream stream)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        TORCH_CHECK(!poisoned_, "QMMA cleanup previously failed; restart the distributed job");
        const aclrtStream waitStream = stream != nullptr ? stream : lastStream_;
        if (waitStream != nullptr) {
            TORCH_CHECK(aclrtSynchronizeStream(waitStream) == ACL_SUCCESS, "failed to drain the QMMA stream");
        }
        if (symmetricMemory_ != nullptr || barrierWorkspace_ != nullptr) {
            aclshmem_barrier_all();
        }
        if (barrierWorkspace_ != nullptr) {
            aclshmem_free(barrierWorkspace_);
            barrierWorkspace_ = nullptr;
        }
        if (symmetricMemory_ != nullptr) {
            aclshmem_free(symmetricMemory_);
            symmetricMemory_ = nullptr;
        }
        TORCH_CHECK(aclshmem_finalize() == ACLSHMEM_SUCCESS, "QMMA SHMEM finalization failed");
        Reset();
    }

    void Poison()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        PoisonShmem();
    }

    void CheckDevice(int32_t device) const
    {
        EnsureInitialized();
        TORCH_CHECK(device == device_, "QMMA tensor is not on the initialized NPU");
    }

    void RecordStream(aclrtStream stream)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        if (lastStream_ != nullptr && lastStream_ != stream) {
            TORCH_CHECK(aclrtSynchronizeStream(lastStream_) == ACL_SUCCESS, "failed to drain the previous QMMA stream");
        }
        lastStream_ = stream;
    }

    uint32_t Rank() const
    {
        EnsureInitialized();
        return static_cast<uint32_t>(rank_);
    }

    uint32_t WorldSize() const
    {
        EnsureInitialized();
        return static_cast<uint32_t>(worldSize_);
    }

    uint32_t Lanes() const
    {
        EnsureInitialized();
        return lanes_;
    }

    uint32_t AicNum() const
    {
        EnsureInitialized();
        return aicNum_;
    }

    uint64_t HeapBytes() const
    {
        EnsureInitialized();
        return heapBytes_;
    }

    uint64_t FftsAddress() const
    {
        EnsureInitialized();
        return fftsAddress_;
    }

    aclrtStream Stream() const
    {
        EnsureInitialized();
        return lastStream_;
    }

    uint8_t* EnsureWorkspace(uint64_t bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        const uint64_t aligned = AlignUp(bytes, 4096U);
        if (symmetricMemory_ != nullptr) {
            TORCH_CHECK(aligned <= symmetricBytes_, "QMMA shape exceeds the allocated symmetric workspace");
            return symmetricMemory_;
        }
        TORCH_CHECK(aligned <= heapBytes_, "QMMA symmetric workspace exceeds heap_size");
        void* memory = aclshmem_align(64U, aligned);
        TORCH_CHECK(memory != nullptr, "QMMA symmetric allocation failed: ", aligned);
        symmetricMemory_ = static_cast<uint8_t*>(memory);
        symmetricBytes_ = aligned;
        return symmetricMemory_;
    }

    uint64_t EnsureBarrierWorkspace(uint64_t barrierBytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        TORCH_CHECK(barrierBytes > 0, "QMMA barrier workspace must be positive");
        if (barrierWorkspace_ == nullptr) {
            barrierWorkspace_ = static_cast<uint8_t*>(aclshmem_calloc(1, barrierBytes));
            TORCH_CHECK(barrierWorkspace_ != nullptr, "QMMA barrier workspace allocation failed");
            barrierBytes_ = barrierBytes;
        }
        TORCH_CHECK(barrierBytes <= barrierBytes_, "QMMA barrier workspace is too small");
        return barrierBytes_;
    }

    void InitializeControl(uint8_t* symmetric, uint64_t controlOffset, uint64_t controlBytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        TORCH_CHECK(symmetric == symmetricMemory_ && controlOffset <= symmetricBytes_,
                    "QMMA control region exceeds the symmetric workspace");
        TORCH_CHECK(controlBytes <= symmetricBytes_ - controlOffset,
                    "QMMA control region size disagrees with CatCCOS layout");
        TORCH_CHECK(barrierWorkspace_ != nullptr, "QMMA barrier workspace is not allocated");
        TORCH_CHECK(aclrtMemset(symmetric + controlOffset, controlBytes, 0, controlBytes) == ACL_SUCCESS,
                    "QMMA control-area initialization failed");
        TORCH_CHECK(barrierEpoch_ != std::numeric_limits<uint32_t>::max(), "QMMA barrier epoch exhausted");
        const int32_t status = LaunchHierarchicalBarrier(Rank(), worldSize_, kRanksPerNode, ++barrierEpoch_,
                                                         barrierWorkspace_, lastStream_, fftsAddress_);
        TORCH_CHECK(status == 0, "QMMA initial barrier failed: ", status);
    }

    void FinalBarrier()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        TORCH_CHECK(barrierEpoch_ != std::numeric_limits<uint32_t>::max(), "QMMA barrier epoch exhausted");
        TORCH_CHECK(LaunchHierarchicalBarrier(Rank(), WorldSize(), kRanksPerNode, ++barrierEpoch_, barrierWorkspace_,
                                              lastStream_, fftsAddress_) == 0,
                    "QMMA final barrier failed");
    }

    uint32_t NextEpoch()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EnsureInitialized();
        TORCH_CHECK(epoch_ != std::numeric_limits<uint32_t>::max(), "QMMA epoch exhausted");
        return ++epoch_;
    }

private:
    QmmaRuntime() = default;
    QmmaRuntime(const QmmaRuntime&) = delete;
    QmmaRuntime& operator=(const QmmaRuntime&) = delete;

    void EnsureInitialized() const
    {
        TORCH_CHECK(initialized_ && !poisoned_, "QMMA is not initialized; call init_qmma_group() first");
    }

    void InitializeShmem()
    {
        const bool crossNode = worldSize_ > kRanksPerNode;
        TORCH_CHECK(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, lanes_) == ACLSHMEM_SUCCESS,
                    "failed to configure QMMA UDMA QPs");
        if (crossNode) {
            TORCH_CHECK(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, lanes_) == ACLSHMEM_SUCCESS,
                        "failed to configure QMMA RoCE QPs");
        }
        aclshmemx_init_attr_t attributes{};
        auto mode = ACLSHMEMX_INIT_WITH_DEFAULT;
        if (!uid_.empty()) {
            aclshmemx_uniqueid_t uniqueId{};
            std::memcpy(&uniqueId, uid_.data(), sizeof(uniqueId));
            TORCH_CHECK(aclshmemx_set_attr_uniqueid_args(rank_, worldSize_, static_cast<int64_t>(heapBytes_), &uniqueId,
                                                         &attributes) == ACLSHMEM_SUCCESS,
                        "failed to configure the QMMA SHMEM unique ID");
            mode = ACLSHMEMX_INIT_WITH_UNIQUEID;
        } else {
            aclshmemx_uniqueid_t ignored{};
            TORCH_CHECK(
                set_attr(rank_, worldSize_, heapBytes_, endpoint_.c_str(), &attributes, &ignored) == ACLSHMEM_SUCCESS,
                "failed to configure QMMA SHMEM attributes");
        }
        attributes.option_attr.data_op_engine_type =
            crossNode ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_ROCE)
                      : ACLSHMEM_DATA_OP_UDMA;
        attributes.option_attr.shm_init_timeout = kDefaultTimeout;
        attributes.option_attr.shm_create_timeout = kDefaultTimeout;
        attributes.option_attr.control_operation_timeout = kDefaultTimeout;
        attributes.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        runtimeStarted_ = true;
        TORCH_CHECK(aclshmemx_init_attr(mode, &attributes) == ACLSHMEM_SUCCESS, "QMMA SHMEM initialization failed");
        fftsAddress_ = shmemx_get_ffts_config();
        for (int32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer == rank_ ||
                peer / static_cast<int32_t>(kRanksPerNode) != rank_ / static_cast<int32_t>(kRanksPerNode)) {
                continue;
            }
            for (uint32_t qp = 0; qp < lanes_; ++qp) {
                aclshmemx_net_layer_t layer{};
                const int status =
                    aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, static_cast<int>(qp), &layer);
                TORCH_CHECK(status == ACLSHMEM_SUCCESS, "QMMA rank ", rank_, " has no UDMA route to local peer ", peer,
                            " QP ", qp, ": ", status);
            }
        }
    }

    void PoisonShmem()
    {
        if (barrierWorkspace_ != nullptr) {
            aclshmem_free(barrierWorkspace_);
            barrierWorkspace_ = nullptr;
        }
        if (symmetricMemory_ != nullptr) {
            aclshmem_free(symmetricMemory_);
            symmetricMemory_ = nullptr;
        }
        if (runtimeStarted_) {
            aclshmem_finalize();
            runtimeStarted_ = false;
        }
        Reset();
        poisoned_ = true;
    }

    void Reset()
    {
        initialized_ = false;
        initializing_ = false;
        poisoned_ = false;
        runtimeStarted_ = false;
        rank_ = -1;
        worldSize_ = 0;
        device_ = -1;
        lanes_ = 1;
        aicNum_ = 0;
        heapBytes_ = 0;
        symmetricBytes_ = 0;
        barrierBytes_ = 0;
        symmetricMemory_ = nullptr;
        barrierWorkspace_ = nullptr;
        lastStream_ = nullptr;
        fftsAddress_ = 0;
        epoch_ = 0;
        barrierEpoch_ = 0;
        endpoint_.clear();
        uid_.clear();
    }

    mutable std::mutex mutex_;
    bool initialized_{false};
    bool initializing_{false};
    bool poisoned_{false};
    bool runtimeStarted_{false};
    int32_t rank_{-1};
    int32_t worldSize_{0};
    int32_t device_{-1};
    uint32_t lanes_{1};
    uint32_t aicNum_{0};
    uint64_t heapBytes_{0};
    uint64_t symmetricBytes_{0};
    uint64_t barrierBytes_{0};
    uint8_t* symmetricMemory_{nullptr};
    uint8_t* barrierWorkspace_{nullptr};
    aclrtStream lastStream_{nullptr};
    uint64_t fftsAddress_{0};
    uint32_t epoch_{0};
    uint32_t barrierEpoch_{0};
    std::string endpoint_;
    std::vector<uint8_t> uid_;
};

uint32_t CheckedUint32(int64_t value, const char* name)
{
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<uint32_t>::max(), name, " must fit uint32");
    return static_cast<uint32_t>(value);
}

ProblemShape InferShape(const at::Tensor& a, const at::Tensor& b, int64_t format, int64_t epSize)
{
    TORCH_CHECK(format >= 0 && format <= 2, "format must be 0 (MXFP8), 1 (MXFP4), or 2 (BF16)");
    TORCH_CHECK(epSize >= 2 && epSize <= kMaxWorldSize, "ep_size must be in [2, ", kMaxWorldSize, "]");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be two-dimensional");
    TORCH_CHECK(a.size(1) > 0 && b.size(1) > 0, "packed input widths must be positive");
    const int64_t k = format == 1 ? a.size(1) * 2 : a.size(1);
    const int64_t n = format == 1 ? b.size(1) * 2 : b.size(1);
    TORCH_CHECK(b.size(0) == k, "b's first dimension must equal k");
    return {CheckedUint32(a.size(0), "m"), CheckedUint32(n, "n"), CheckedUint32(k, "k")};
}

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined() && tensor.device().type() == c10::DeviceType::PrivateUse1, name,
                " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckInputs(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>& aScale,
                 const c10::optional<at::Tensor>& bScale, int64_t format, const ProblemShape& shape, int64_t epSize)
{
    CheckNpuTensor(a, "a");
    CheckNpuTensor(b, "b");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same NPU");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");
    if (format == 2) {
        TORCH_CHECK(a.scalar_type() == at::kBFloat16, "BF16 format requires BF16 inputs");
        TORCH_CHECK(!aScale.has_value() && !bScale.has_value(), "BF16 format does not take MX scales");
    } else {
        TORCH_CHECK(a.element_size() == 1 && b.element_size() == 1, "MX inputs must use one-byte physical storage");
        TORCH_CHECK(aScale.has_value() && bScale.has_value(), "MX formats require both scales");
        CheckNpuTensor(aScale.value(), "a_scale");
        CheckNpuTensor(bScale.value(), "b_scale");
        TORCH_CHECK(aScale->device() == a.device() && bScale->device() == a.device(),
                    "scales and inputs must be on the same NPU");
        const uint32_t groups = (shape.k + 31U) / 32U;
        const uint32_t paddedGroups = (groups + 1U) / 2U * 2U;
        TORCH_CHECK(aScale->dim() == 2 && aScale->size(0) == shape.m && aScale->size(1) == paddedGroups,
                    "a_scale must have shape [m, padded_groups]");
        TORCH_CHECK(bScale->dim() == 3 && bScale->size(0) == paddedGroups / 2 && bScale->size(1) == shape.n &&
                        bScale->size(2) == 2,
                    "b_scale must have shape [padded_groups/2, n, 2]");
    }
    TORCH_CHECK(shape.k % 32 == 0, "k must be divisible by 32");
    const int64_t tileN = format == 2 ? 128 : 256;
    TORCH_CHECK(shape.n % epSize == 0 && (shape.n / epSize) % tileN == 0, "n/ep_size must be divisible by ", tileN);
}

void CheckOutput(const at::Tensor& output, const at::Tensor& a, const ProblemShape& shape, int64_t epSize)
{
    CheckNpuTensor(output, "out");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16 && output.device() == a.device(),
                "out must be contiguous BF16 on the input NPU");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == static_cast<int64_t>(shape.m) * epSize &&
                    output.size(1) == static_cast<int64_t>(shape.n) / epSize,
                "out must have shape [ep_size*m, n/ep_size]");
}

uint32_t ResolveInterval(uint32_t m, int64_t requested)
{
    TORCH_CHECK(requested >= 0 && requested <= std::numeric_limits<uint32_t>::max(),
                "comm_interval must be a non-negative uint32");
    return requested == 0 ? std::min<uint32_t>(32, std::max<uint32_t>(1, (m + 255U) / 256U))
                          : static_cast<uint32_t>(requested);
}

const char* Validate(uint32_t format, const ProblemShape& shape, uint32_t worldSize, uint32_t lanes, uint32_t interval,
                     uint32_t aicNum, uint64_t heapBytes)
{
    if (format > 2 || worldSize < 2 || worldSize > kMaxWorldSize || lanes < 1 || lanes > 4 || aicNum == 0 ||
        aicNum > 64 || 2ULL * aicNum < worldSize) {
        return "invalid format, rank count, lane count, or AIC count";
    }
    const uint32_t tileN = format == 2 ? 128U : 256U;
    if (shape.m == 0 || shape.n == 0 || shape.k == 0 || shape.k % 32 != 0 || shape.n % worldSize != 0 ||
        (shape.n / worldSize) % tileN != 0) {
        return "invalid M/N/K or N/world_size alignment";
    }
    const QmmaLaunchArgs args = MakeLaunchArgs(shape, format, worldSize, interval, 0, lanes, aicNum, 0);
    if (interval == 0 || interval > std::numeric_limits<uint32_t>::max() / args.tile_m) {
        return "invalid comm_interval";
    }
    return SymmetricBytes(&args) <= heapBytes ? nullptr : "shape exceeds heap_size";
}

void InitializeQmma(const at::Tensor& deviceAnchor, int64_t rank, int64_t worldSize, int64_t lanes, int64_t heapSize,
                    const std::string& endpoint, const std::vector<int64_t>& uid)
{
    CheckNpuTensor(deviceAnchor, "device_anchor");
    TORCH_CHECK(worldSize >= 2 && worldSize <= kMaxWorldSize && rank >= 0 && rank < worldSize, "invalid QMMA topology");
    TORCH_CHECK(lanes >= 1 && lanes <= 4 && heapSize > 0, "invalid QMMA lanes or heap_size");
    TORCH_CHECK(uid.empty() || uid.size() == static_cast<int64_t>(QmmaRuntime::UniqueIdSize()),
                "invalid QMMA SHMEM unique ID");
    std::vector<uint8_t> bytes;
    bytes.reserve(uid.size());
    for (int64_t value : uid) {
        TORCH_CHECK(value >= 0 && value <= 255, "QMMA unique ID byte out of range");
        bytes.push_back(static_cast<uint8_t>(value));
    }
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    QmmaRuntime::Instance().Initialize(static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
                                       static_cast<uint32_t>(lanes), static_cast<uint64_t>(heapSize), endpoint, bytes);
}

void DestroyQmma(const at::Tensor& deviceAnchor)
{
    CheckNpuTensor(deviceAnchor, "device_anchor");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    QmmaRuntime::Instance().Destroy(stream);
}

void PoisonQmma()
{
    QmmaRuntime::Instance().Poison();
}

at::Tensor QuantMatmulAllToAllImpl(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>& aScale,
                                   const c10::optional<at::Tensor>& bScale, int64_t format, int64_t commInterval,
                                   int64_t epSize, const c10::optional<at::Tensor>& outputOptional)
{
    const ProblemShape shape = InferShape(a, b, format, epSize);
    const c10::OptionalDeviceGuard guard(a.device());
    CheckInputs(a, b, aScale, bScale, format, shape, epSize);
    QmmaRuntime& runtime = QmmaRuntime::Instance();
    runtime.CheckDevice(a.device().index());
    TORCH_CHECK(runtime.WorldSize() == epSize, "ep_size must equal the initialized QMMA world_size");

    at::Tensor output =
        outputOptional.has_value()
            ? outputOptional.value()
            : at::empty({static_cast<int64_t>(shape.m) * epSize, static_cast<int64_t>(shape.n) / epSize},
                        a.options().dtype(at::kBFloat16));
    CheckOutput(output, a, shape, epSize);
    TORCH_CHECK(!output.is_same(a) && !output.is_same(b), "QMMA output must not alias an input");
    if (aScale.has_value()) {
        TORCH_CHECK(!output.is_same(aScale.value()), "QMMA output must not alias a_scale");
    }
    if (bScale.has_value()) {
        TORCH_CHECK(!output.is_same(bScale.value()), "QMMA output must not alias b_scale");
    }

    const uint32_t interval = ResolveInterval(shape.m, commInterval);
    const char* error = Validate(static_cast<uint32_t>(format), shape, runtime.WorldSize(), runtime.Lanes(), interval,
                                 runtime.AicNum(), runtime.HeapBytes());
    TORCH_CHECK(error == nullptr, "invalid QuantMatmulAllToAll configuration: ", error);

    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "failed to acquire the current NPU stream");
    QmmaLaunchArgs layoutArgs =
        MakeLaunchArgs(shape, format, epSize, interval, runtime.Rank(), runtime.Lanes(), runtime.AicNum(), 0);
    const uint64_t symmetricBytes = SymmetricBytes(&layoutArgs);
    const uint64_t barrierBytes = BarrierWorkspaceBytes(runtime.WorldSize(), kRanksPerNode);
    runtime.RecordStream(stream);
    uint8_t* symmetric = runtime.EnsureWorkspace(symmetricBytes);
    runtime.EnsureBarrierWorkspace(barrierBytes);
    runtime.InitializeControl(symmetric, ControlRegionOffset(&layoutArgs), ControlRegionBytes(&layoutArgs));
    QmmaLaunchArgs args = MakeLaunchArgs(
        shape, format, epSize, interval, runtime.Rank(), runtime.Lanes(), runtime.AicNum(), runtime.NextEpoch(),
        symmetric, a.data_ptr(), b.data_ptr(), aScale.has_value() ? aScale->data_ptr() : nullptr,
        bScale.has_value() ? bScale->data_ptr() : nullptr, output.data_ptr(), stream, runtime.FftsAddress());
    const int32_t status = LaunchQuantMatmulAllToAll(&args);
    TORCH_CHECK(status == 0, "CatCCOS QMMA launch failed: ", status);

    const uint64_t outputOffset = OutputOffset(&layoutArgs);
    const uint64_t outputBytes = static_cast<uint64_t>(shape.m) * shape.n * sizeof(uint16_t);
    TORCH_CHECK(aclrtMemcpyAsync(output.data_ptr(), outputBytes, symmetric + outputOffset, outputBytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream) == ACL_SUCCESS,
                "failed to copy the QMMA symmetric output");
    TORCH_CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "failed to complete the QMMA output copy");
    runtime.FinalBarrier();
    return output;
}

at::Tensor QuantMatmulAllToAllNpu(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>& aScale,
                                  const c10::optional<at::Tensor>& bScale, int64_t format, int64_t commInterval,
                                  int64_t epSize)
{
    return QuantMatmulAllToAllImpl(a, b, aScale, bScale, format, commInterval, epSize, c10::nullopt);
}

at::Tensor QuantMatmulAllToAllOutNpu(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>& aScale,
                                     const c10::optional<at::Tensor>& bScale, int64_t format, int64_t commInterval,
                                     int64_t epSize, const at::Tensor& output)
{
    return QuantMatmulAllToAllImpl(a, b, aScale, bScale, format, commInterval, epSize, output);
}

at::Tensor QuantMatmulAllToAllMeta(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>&,
                                   const c10::optional<at::Tensor>&, int64_t format, int64_t, int64_t epSize)
{
    const ProblemShape shape = InferShape(a, b, format, epSize);
    return at::empty({static_cast<int64_t>(shape.m) * epSize, static_cast<int64_t>(shape.n) / epSize},
                     a.options().dtype(at::kBFloat16));
}

at::Tensor QuantMatmulAllToAllOutMeta(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>&,
                                      const c10::optional<at::Tensor>&, int64_t format, int64_t, int64_t epSize,
                                      const at::Tensor& output)
{
    const ProblemShape shape = InferShape(a, b, format, epSize);
    TORCH_CHECK(output.dim() == 2 && output.size(0) == static_cast<int64_t>(shape.m) * epSize &&
                    output.size(1) == static_cast<int64_t>(shape.n) / epSize && output.scalar_type() == at::kBFloat16,
                "out must have shape [ep_size*m, n/ep_size]");
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "_initialize_qmma(Tensor device_anchor, int rank, int world_size, int lane_num, int heap_size, str endpoint, "
        "int[] uid) -> ()",
        &InitializeQmma);
    m.def("_qmma_unique_id_size() -> int", &QmmaRuntime::UniqueIdSize);
    m.def("_qmma_unique_id() -> int[]", &QmmaRuntime::UniqueId);
    m.def("_destroy_qmma(Tensor device_anchor) -> ()", &DestroyQmma);
    m.def("_poison_qmma() -> ()", &PoisonQmma);
    m.def(
        "quant_matmul_alltoall(Tensor a, Tensor b, Tensor? a_scale, Tensor? b_scale, int format, int comm_interval, "
        "int ep_size) -> Tensor");
    m.def(
        "quant_matmul_alltoall_out(Tensor a, Tensor b, Tensor? a_scale, Tensor? b_scale, int format, int "
        "comm_interval, int ep_size, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("quant_matmul_alltoall", QuantMatmulAllToAllMeta);
    m.impl("quant_matmul_alltoall_out", QuantMatmulAllToAllOutMeta);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("quant_matmul_alltoall", QuantMatmulAllToAllNpu);
    m.impl("quant_matmul_alltoall_out", QuantMatmulAllToAllOutNpu);
}

}  // namespace deepep::ep_gmm_fused
