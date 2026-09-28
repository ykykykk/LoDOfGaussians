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
                            int64_t elements, int64_t width, int64_t capacity, int64_t raw_stride,
                            float correction1, float correction2_sqrt) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int64_t row = i / width, col = i % width, slot = slots[row];
    if (slot < 0 || slot >= capacity) return;
    const int64_t base = slot * (3 * width) + col;
    const int64_t raw_index = row * raw_stride + col;
    const float g = frozen[row] ? 0.f : grad[i];
    const float m = state[base + width] * 0.9f + g * 0.1f;
    const float v = state[base + 2 * width] * 0.999f + g * g * 0.001f;
    const float denominator = sqrtf(v) / correction2_sqrt + 1e-8f;
    const float p = raw[raw_index] - (m / denominator) * (rates[col] / correction1);
    state[base] = raw[raw_index] = p;
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
    CUDA_CONTIG(grad); CUDA_CONTIG(rates); CUDA_CONTIG(frozen);
    FP32(raw); FP32(grad); FP32(rates);
    const int64_t d = state.size(1) / 3, n = slots.numel();
    TORCH_CHECK(raw.dim() == 2 && raw.size(0) == n && raw.size(1) == d && grad.sizes() == raw.sizes(), "invalid raw/gradient dimensions");
    // Overflow packets expose parameters as a strided view of [p | m | v].
    TORCH_CHECK(raw.is_cuda() && raw.stride(1) == 1 && raw.stride(0) >= d, "invalid raw strides");
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
            n*d, d, state.size(0), raw.stride(0), float(correction1), float(correction2_sqrt));
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

// Exact four-plane sphere test; no LoD substitution or point budget.
__global__ void flat_visible_kernel(const float* bounds, const float* planes, bool* mask, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool keep = true;
    for (int p = 0; p < 4; ++p) {
        const float d = bounds[i*4] * planes[p*4] + bounds[i*4+1] * planes[p*4+1]
                      + bounds[i*4+2] * planes[p*4+2] + planes[p*4+3];
        keep = keep && (d + bounds[i*4+3] >= 0.f);
    }
    mask[i] = keep;
}

__global__ void update_bounds_kernel(float* bounds, const int64_t* ids, const float* raw,
                                     int64_t n, int64_t capacity, int64_t stride) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t id = ids[i];
    if (id < 0 || id >= capacity) return;
    const float* r = raw + i * stride;
    bounds[id*4] = r[0]; bounds[id*4+1] = r[1]; bounds[id*4+2] = r[2];
    // Preserve Torch amax NaN propagation (fmaxf alone suppresses NaNs).
    const float largest = (isnan(r[3]) || isnan(r[4]) || isnan(r[5]))
        ? nanf("") : fmaxf(r[3], fmaxf(r[4], r[5]));
    bounds[id*4+3] = 3.f * expf(largest);
}

static void check_bounds(const torch::Tensor& bounds) {
    CUDA_CONTIG(bounds); FP32(bounds);
    TORCH_CHECK(bounds.dim() == 2 && bounds.size(1) == 4, "bounds must be FP32 [N,4]");
}

torch::Tensor flat_visible_cuda(torch::Tensor bounds, torch::Tensor planes) {
    check_bounds(bounds); CUDA_CONTIG(planes); FP32(planes);
    TORCH_CHECK(planes.dim() == 2 && planes.size(0) == 4 && planes.size(1) == 4, "planes must be [4,4]");
    TORCH_CHECK(planes.device() == bounds.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(bounds.device());
    const int64_t n = bounds.size(0);
    auto mask = torch::empty({n}, bounds.options().dtype(torch::kBool));
    if (n) {
        flat_visible_kernel<<<(n+255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            bounds.data_ptr<float>(), planes.data_ptr<float>(), mask.data_ptr<bool>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return mask;
}

void update_bounds_cuda(torch::Tensor bounds, torch::Tensor ids, torch::Tensor raw) {
    check_bounds(bounds); CUDA_CONTIG(ids); FP32(raw);
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == torch::kInt64, "ids must be int64 [N]");
    TORCH_CHECK(raw.is_cuda() && raw.dim() == 2 && raw.size(0) == ids.numel()
                && raw.size(1) >= 6 && raw.stride(1) == 1 && raw.stride(0) >= raw.size(1), "invalid raw layout");
    TORCH_CHECK(ids.device() == bounds.device() && raw.device() == bounds.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(bounds.device());
    const int64_t n = ids.numel();
    if (n) {
        update_bounds_kernel<<<(n+255)/256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            bounds.data_ptr<float>(), ids.data_ptr<int64_t>(), raw.data_ptr<float>(), n, bounds.size(0), raw.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

__global__ void paged_visible_kernel(const float* state, const int64_t* counts,
    const bool* requested, const bool* sky, int64_t block_rows, const float* planes,
    bool* mask, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t page = i / block_rows;
    if (!requested[page] || i % block_rows >= counts[page]) { mask[i] = false; return; }
    if (sky[page]) { mask[i] = true; return; }
    const float* r = state + i * 69;
    const float largest = (isnan(r[3]) || isnan(r[4]) || isnan(r[5]))
        ? nanf("") : fmaxf(r[3], fmaxf(r[4], r[5]));
    const float radius = 3.f * expf(largest);
    bool keep = true;
    for (int p = 0; p < 4; ++p) {
        const float d = r[0]*planes[p*4] + r[1]*planes[p*4+1]
                      + r[2]*planes[p*4+2] + planes[p*4+3];
        keep = keep && (d + radius >= 0.f);
    }
    mask[i] = keep;
}

__global__ void recompute_page_bounds_kernel(const float* state, const int64_t* counts,
    const int64_t* mapping, const bool* requested, int64_t block_rows, float* bounds, int64_t blocks) {
    const int64_t page = blockIdx.x, bid = mapping[page];
    if (!requested[page] || bid < 0 || bid >= blocks || counts[page] <= 0) return;
    __shared__ float work[6][256];
    float lo[3] = {INFINITY, INFINITY, INFINITY};
    float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int64_t row=threadIdx.x; row<counts[page] && row<block_rows; row+=blockDim.x) {
        const float* r=state+(page*block_rows+row)*69;
        const float radius=3.f*expf(fmaxf(r[3],fmaxf(r[4],r[5])));
        for(int axis=0;axis<3;++axis) {
            lo[axis]=fminf(lo[axis],r[axis]-radius);
            hi[axis]=fmaxf(hi[axis],r[axis]+radius);
        }
    }
    for(int axis=0;axis<3;++axis) {
        work[axis][threadIdx.x]=lo[axis];
        work[axis+3][threadIdx.x]=hi[axis];
    }
    __syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(threadIdx.x<stride) for(int axis=0;axis<3;++axis) {
            work[axis][threadIdx.x]=fminf(work[axis][threadIdx.x],work[axis][threadIdx.x+stride]);
            work[axis+3][threadIdx.x]=fmaxf(work[axis+3][threadIdx.x],work[axis+3][threadIdx.x+stride]);
        }
        __syncthreads();
    }
    if(threadIdx.x==0) for(int axis=0;axis<6;++axis) bounds[bid*6+axis]=work[axis][0];
}

torch::Tensor paged_visible_cuda(torch::Tensor state, torch::Tensor counts, torch::Tensor requested,
    torch::Tensor sky, int64_t block_rows, torch::Tensor planes) {
    CUDA_CONTIG(state); CUDA_CONTIG(counts); CUDA_CONTIG(requested); CUDA_CONTIG(sky);
    CUDA_CONTIG(planes); FP32(state); FP32(planes);
    TORCH_CHECK(state.dim()==2 && state.size(1)==69 && block_rows>0 && state.size(0)%block_rows==0, "invalid paged state");
    const int64_t pages=state.size(0)/block_rows;
    TORCH_CHECK(counts.dim()==1 && counts.numel()==pages && counts.scalar_type()==torch::kInt64, "invalid page counts");
    TORCH_CHECK(requested.dim()==1 && requested.numel()==pages && requested.scalar_type()==torch::kBool, "invalid requested pages");
    TORCH_CHECK(sky.dim()==1 && sky.numel()==pages && sky.scalar_type()==torch::kBool, "invalid sky pages");
    TORCH_CHECK(planes.dim()==2 && planes.size(0)==4 && planes.size(1)==4, "planes must be [4,4]");
    TORCH_CHECK(counts.device()==state.device() && requested.device()==state.device()
        && sky.device()==state.device() && planes.device()==state.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(state.device());
    auto result=torch::empty({state.size(0)},state.options().dtype(torch::kBool));
    if (state.size(0)) {
        paged_visible_kernel<<<(state.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
            state.data_ptr<float>(),counts.data_ptr<int64_t>(),requested.data_ptr<bool>(),sky.data_ptr<bool>(),
            block_rows,planes.data_ptr<float>(),result.data_ptr<bool>(),state.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return result;
}

void recompute_page_bounds_cuda(torch::Tensor state, torch::Tensor counts, torch::Tensor mapping,
    torch::Tensor requested, int64_t block_rows, torch::Tensor bounds) {
    CUDA_CONTIG(state); CUDA_CONTIG(counts); CUDA_CONTIG(mapping); CUDA_CONTIG(requested); CUDA_CONTIG(bounds);
    FP32(state); FP32(bounds);
    TORCH_CHECK(state.dim()==2 && state.size(1)==69 && block_rows>0 && state.size(0)%block_rows==0, "invalid paged state");
    const int64_t pages=state.size(0)/block_rows;
    TORCH_CHECK(counts.dim()==1 && counts.numel()==pages && counts.scalar_type()==torch::kInt64, "invalid page counts");
    TORCH_CHECK(mapping.dim()==1 && mapping.numel()==pages && mapping.scalar_type()==torch::kInt64, "invalid page mapping");
    TORCH_CHECK(requested.dim()==1 && requested.numel()==pages && requested.scalar_type()==torch::kBool, "invalid requested pages");
    TORCH_CHECK(bounds.dim()==2 && bounds.size(1)==6, "invalid page bounds");
    TORCH_CHECK(counts.device()==state.device() && mapping.device()==state.device()
        && requested.device()==state.device() && bounds.device()==state.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(state.device());
    if(pages) {
        recompute_page_bounds_kernel<<<pages,256,0,at::cuda::getCurrentCUDAStream()>>>(
            state.data_ptr<float>(),counts.data_ptr<int64_t>(),mapping.data_ptr<int64_t>(),requested.data_ptr<bool>(),
            block_rows,bounds.data_ptr<float>(),bounds.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
