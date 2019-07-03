#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace {
template <typename scalar_t>
__global__ void cgm_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int32_t group_size,
    const int32_t B,
    const int32_t C,
    const int32_t W,
    const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t size = B*C*W*H;
  const int32_t CWH = C*W*H;
  const int32_t WH = W*H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b*CWH) / WH;
  const int32_t w = (idx - b*CWH - c*WH) / W;
  const int32_t h = idx - b*CWH - c*WH - w*H;
  const int32_t base_offset = b*CWH + w*W + h;
  int32_t gidx;

  if (c < (C / group_size)) {
    gidx = c*group_size*WH + base_offset;
    int32_t group_max_idx = -1;
    scalar_t group_max_val = 0;
    for (int i = 0; i < group_size; ++i) {
      gidx = (c*group_size + i)*WH + base_offset;
      if (input[gidx] > group_max_val) {
        group_max_idx = i;
        group_max_val = input[gidx];
      }
    }
    if (group_max_idx != -1) {
      gidx = (c*group_size + group_max_idx)*WH + base_offset;
      output[gidx] = group_max_val;
    }
  }
}

template <typename scalar_t>
__global__ void cgm_cuda_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ input, 
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size && input[idx] <= 0) grad_input[idx] = 0;
}
} // namespace

at::Tensor cgm_cuda_forward(
    const at::Tensor input,
    const int32_t group_size) {
  const auto ndim = input.ndimension();
  const auto B = input.size(0);
  const auto C = input.size(1);
  auto W = 1;
  auto H = 1;
  if (ndim == 4) {
    W = input.size(2);
    H = input.size(3);
  }
  const auto size = B*C*W*H;
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cgm_forward_cuda", ([&] {
    cgm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        group_size,
        B,
        C,
        W,
        H);
  }));

  return output;
}

at::Tensor cgm_cuda_backward(
    at::Tensor grad_input,
    const at::Tensor input) {
  const auto size = grad_input.numel();
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "shift_backward_cuda", ([&] {
    cgm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_input.data<scalar_t>(),
        input.data<scalar_t>(),
        size);
  }));

  return grad_input;
}
