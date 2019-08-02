#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void static_cgm_cuda_forward_kernel(
    const scalar_t *__restrict__ input, scalar_t *__restrict__ output,
    const int32_t *__restrict__ groups, const int32_t num_groups,
    const int32_t group_size, const int32_t B, const int32_t C, const int32_t W,
    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t size = B * C * W * H;
  const int32_t CWH = C * W * H;
  const int32_t WH = W * H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b * CWH) / WH;
  const int32_t w = (idx - b * CWH - c * WH) / W;
  const int32_t h = idx - b * CWH - c * WH - w * H;
  const int32_t base_offset = b * CWH + w * W + h;
  int32_t gidx;

  if (c < num_groups) {
    gidx = c * group_size * WH + base_offset;
    int32_t group_max_idx = -1;
    scalar_t group_max_val = 0;
    for (int i = 0; i < group_size; ++i) {
      int32_t channel_idx = groups[c * group_size + i];
      if (channel_idx < 0)
        continue;
      gidx = channel_idx * WH + base_offset;
      if (input[gidx] > group_max_val) {
        group_max_idx = i;
        group_max_val = input[gidx];
      }
    }
    if (group_max_idx != -1) {
      int32_t channel_idx = groups[c * group_size + group_max_idx];
      gidx = channel_idx * WH + base_offset;
      output[gidx] = group_max_val;
    }
  }
}

template <typename scalar_t>
__global__ void qpoint_quantize_cuda_forward_kernel(
    const scalar_t *__restrict__ input, scalar_t *__restrict__ output,
    const scalar_t *__restrict__ qpoints, const int32_t num_qpoints,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int32_t min_idx = -1;
    scalar_t min_diff = 1e10;
    const scalar_t x = input[idx];
    for (int i = 0; i < num_qpoints; ++i) {
      scalar_t diff = abs(x - qpoints[i]);
      if (diff < min_diff) {
        min_diff = diff;
        min_idx = i;
      }
    }
    output[idx] = qpoints[min_idx];
  }
}

template <typename scalar_t>
__global__ void
cgm_cuda_forward_kernel(const scalar_t *__restrict__ input,
                        scalar_t *__restrict__ output, const int32_t group_size,
                        const float max_clamp, const int32_t B, const int32_t C,
                        const int32_t W, const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t size = B * C * W * H;
  const int32_t CWH = C * W * H;
  const int32_t WH = W * H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b * CWH) / WH;
  const int32_t w = (idx - b * CWH - c * WH) / W;
  const int32_t h = idx - b * CWH - c * WH - w * H;
  const int32_t base_offset = b * CWH + w * W + h;
  int32_t gidx;

  if (c < (C / group_size)) {
    gidx = c * group_size * WH + base_offset;
    int32_t group_max_idx = -1;
    scalar_t group_max_val = 0;
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      if (input[gidx] > group_max_val) {
        group_max_idx = i;
        group_max_val = input[gidx];
      }

      // keep values larger than max_clamp
      if (input[gidx] > max_clamp) {
        output[gidx] = input[gidx];
      }
    }
    if (group_max_idx != -1) {
      gidx = (c * group_size + group_max_idx) * WH + base_offset;
      output[gidx] = group_max_val;
    }
  }
}

template <typename scalar_t>
__global__ void cgm_cuda_backward_kernel(scalar_t *__restrict__ grad_input,
                                         const scalar_t *__restrict__ input,
                                         const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size && input[idx] <= 0)
    grad_input[idx] = 0;
}
} // namespace

at::Tensor static_cgm_cuda_forward(const at::Tensor input,
                                   const at::Tensor groups) {
  const auto ndim = input.ndimension();
  const auto B = input.size(0);
  const auto C = input.size(1);
  auto W = 1;
  auto H = 1;
  if (ndim == 4) {
    W = input.size(2);
    H = input.size(3);
  }
  const auto size = B * C * W * H;
  const auto num_groups = groups.size(0);
  const auto group_size = groups.size(1);
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "static_cgm_forward_cuda", ([&] {
        static_cgm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), output.data<scalar_t>(),
            groups.data<int32_t>(), num_groups, group_size, B, C, W, H);
      }));

  return output;
}

at::Tensor qpoint_quantize_cuda_forward(const at::Tensor input,
                                        const at::Tensor qpoints) {
  const auto size = input.numel();
  const auto num_qpoints = qpoints.size(0);
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "qpoint_quantize_forward_cuda", ([&] {
        qpoint_quantize_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), output.data<scalar_t>(),
            qpoints.data<scalar_t>(), num_qpoints, size);
      }));

  return output;
}

at::Tensor cgm_cuda_forward(const at::Tensor input, const int32_t group_size,
                            const float max_clamp) {
  const auto ndim = input.ndimension();
  const auto B = input.size(0);
  const auto C = input.size(1);
  auto W = 1;
  auto H = 1;
  if (ndim == 4) {
    W = input.size(2);
    H = input.size(3);
  }
  const auto size = B * C * W * H;
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "cgm_forward_cuda", ([&] {
        cgm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), output.data<scalar_t>(), group_size,
            max_clamp, B, C, W, H);
      }));

  return output;
}

at::Tensor cgm_cuda_backward(at::Tensor grad_input, const at::Tensor input) {
  const auto size = grad_input.numel();
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      grad_input.type(), "cgm_backward_cuda", ([&] {
        cgm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_input.data<scalar_t>(), input.data<scalar_t>(), size);
      }));

  return grad_input;
}
