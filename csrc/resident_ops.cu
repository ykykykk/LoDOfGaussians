#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>

#define CUDA_CONTIG(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")
#define FP32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be FP32")

__global__ void gather_kernel(const float* state, const int64_t* slots, float* raw,
                              int64_t elements, int64_t width, int64_t capacity) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int64_t row = i / width, col = i % width, slot = slots[row];
    if (slot >= 0 && slot < capacity) raw[i] = state[slot * (3 * width) + col];
}

__global__ void adam_kernel(float* state, const int64_t* slots, float* raw,
                            const float* grad, const float* rates, const bool* frozen,
                            int64_t elements, int64_t width, int64_t capacity,
                            float correction1, float correction2_sqrt) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int64_t row = i / width, col = i % width, slot = slots[row];
    if (slot < 0 || slot >= capacity) return;
    const int64_t base = slot * (3 * width) + col;
    const float g = frozen[row] ? 0.f : grad[i];
    const float m = state[base + width] * 0.9f + g * 0.1f;
    const float v = state[base + 2 * width] * 0.999f + g * g * 0.001f;
    const float denominator = sqrtf(v) / correction2_sqrt + 1e-8f;
    const float p = raw[i] - (m / denominator) * (rates[col] / correction1);
    state[base] = raw[i] = p;
    state[base + width] = m;
    state[base + 2 * width] = v;
}

__device__ bool visible(int64_t i, const float* xyz, const float* bounds,
                        const float* planes, bool culling) {
    if (!culling) return true;
    for (int p = 0; p < 4; ++p) {
        const float d = xyz[i * 3] * planes[p * 4] + xyz[i * 3 + 1] * planes[p * 4 + 1]
                      + xyz[i * 3 + 2] * planes[p * 4 + 2] + planes[p * 4 + 3];
        if (d + bounds[i] < 0.f) return false;
    }
    return true;
}

__device__ bool descend(int64_t i, const float* xyz, const float* minimum,
                        const float* center, float multiplier) {
    const float x = center[0] - xyz[i * 3], y = center[1] - xyz[i * 3 + 1];
    const float z = center[2] - xyz[i * 3 + 2];
    return minimum[i] > (x*x + y*y + z*z) * multiplier;
}

__global__ void cut_kernel(const int32_t* nodes, const float* xyz, const float* bounds,
                           const float* minimum, const float* planes, const float* center,
                           bool* mask, int64_t n, float multiplier, bool culling) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask[i] = false;
    if (!visible(i, xyz, bounds, planes, culling)) return;
    if (nodes[i * 6 + 2] > 0 && descend(i, xyz, minimum, center, multiplier)) return;
    // A node is in the cut exactly when every ancestor is visible and descends.
    // Parent chains are validated by the CPU builder; bound the loop anyway.
    int64_t parent = nodes[i * 6 + 1];
    for (int64_t visited = 0; parent >= 0; ++visited) {
        if (parent >= n || visited >= n) return;
        if (!visible(parent, xyz, bounds, planes, culling)
            || !descend(parent, xyz, minimum, center, multiplier)) return;
        parent = nodes[parent * 6 + 1];
    }
    mask[i] = true;
}

static void check_state(const torch::Tensor& state, const torch::Tensor& slots) {
    CUDA_CONTIG(state); CUDA_CONTIG(slots); FP32(state);
    TORCH_CHECK(state.dim() == 2 && state.size(1) > 0 && state.size(1) % 3 == 0, "invalid state layout");
    TORCH_CHECK(slots.dim() == 1 && slots.scalar_type() == torch::kInt64, "slots must be int64 [N]");
    TORCH_CHECK(slots.device() == state.device(), "state/slots device mismatch");
}

torch::Tensor gather_parameters_cuda(torch::Tensor state, torch::Tensor slots) {
    check_state(state, slots);
    c10::cuda::CUDAGuard guard(state.device());
    const int64_t d = state.size(1) / 3, elements = slots.numel() * d;
    auto raw = torch::empty({slots.numel(), d}, state.options());
    if (elements) {
        gather_kernel<<<(elements + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            state.data_ptr<float>(), slots.data_ptr<int64_t>(), raw.data_ptr<float>(),
            elements, d, state.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return raw;
}

void indexed_adam_cuda(torch::Tensor state, torch::Tensor slots, torch::Tensor raw,
                       torch::Tensor grad, torch::Tensor rates, torch::Tensor frozen,
                       double correction1, double correction2_sqrt) {
    check_state(state, slots);
    CUDA_CONTIG(raw); CUDA_CONTIG(grad); CUDA_CONTIG(rates); CUDA_CONTIG(frozen);
    FP32(raw); FP32(grad); FP32(rates);
    const int64_t d = state.size(1) / 3, n = slots.numel();
    TORCH_CHECK(raw.dim() == 2 && raw.size(0) == n && raw.size(1) == d && grad.sizes() == raw.sizes(), "invalid raw/gradient dimensions");
    TORCH_CHECK(rates.dim() == 1 && rates.numel() == d, "invalid learning rates");
    TORCH_CHECK(frozen.dim() == 1 && frozen.numel() == n && frozen.scalar_type() == torch::kBool, "invalid frozen mask");
    TORCH_CHECK(raw.device() == state.device() && grad.device() == state.device()
                && rates.device() == state.device() && frozen.device() == state.device(), "device mismatch");
    TORCH_CHECK(correction1 > 0 && correction2_sqrt > 0, "invalid Adam correction");
    c10::cuda::CUDAGuard guard(state.device());
    if (n) {
        adam_kernel<<<(n*d + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            state.data_ptr<float>(), slots.data_ptr<int64_t>(), raw.data_ptr<float>(),
            grad.data_ptr<float>(), rates.data_ptr<float>(), frozen.data_ptr<bool>(),
            n*d, d, state.size(0), float(correction1), float(correction2_sqrt));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

torch::Tensor upper_cut_cuda(torch::Tensor nodes, torch::Tensor xyz,
                            torch::Tensor bounds, torch::Tensor minimum,
                            torch::Tensor planes, torch::Tensor center,
                            double multiplier, bool culling) {
    CUDA_CONTIG(nodes); CUDA_CONTIG(xyz); CUDA_CONTIG(bounds); CUDA_CONTIG(minimum);
    CUDA_CONTIG(planes); CUDA_CONTIG(center);
    FP32(xyz); FP32(bounds); FP32(minimum); FP32(planes); FP32(center);
    const int64_t n = nodes.size(0);
    TORCH_CHECK(nodes.dim() == 2 && nodes.size(1) == 6 && nodes.scalar_type() == torch::kInt32, "nodes must be int32 [N,6]");
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(0) == n && xyz.size(1) == 3, "xyz must be [N,3]");
    TORCH_CHECK(bounds.numel() == n && minimum.numel() == n && planes.numel() == 16 && center.numel() == 3, "invalid cut inputs");
    TORCH_CHECK(multiplier > 0, "invalid distance multiplier");
    TORCH_CHECK(xyz.device() == nodes.device() && bounds.device() == nodes.device()
                && minimum.device() == nodes.device() && planes.device() == nodes.device()
                && center.device() == nodes.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(nodes.device());
    auto mask = torch::empty({n}, nodes.options().dtype(torch::kBool));
    if (n) {
        cut_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            nodes.data_ptr<int32_t>(), xyz.data_ptr<float>(), bounds.data_ptr<float>(),
            minimum.data_ptr<float>(), planes.data_ptr<float>(), center.data_ptr<float>(),
            mask.data_ptr<bool>(), n, float(multiplier), culling);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return mask;
}
