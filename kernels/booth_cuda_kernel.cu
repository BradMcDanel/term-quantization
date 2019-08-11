#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

#define MAX_GROUP_SIZE 32
#define MAX_BOOTH_SIZE 31
#define MAX_VALUES 32

namespace {
template <typename scalar_t>
__device__ void booth_encode(const scalar_t input, int32_t *__restrict__ terms,
                             int32_t *num_terms, const float sf) {
  int32_t b0, b1;
  int32_t q_val = int32_t(input / sf);
  int32_t sign = input < 0 ? -1 : 1;
  *num_terms = 0;
  for (int i = 0; i < MAX_BOOTH_SIZE; i++) {
    terms[i] = 0;
  }

  for (int i = MAX_BOOTH_SIZE - 1; i >= 0; i--) {
    if (i == 0) {
      b0 = 0;
    } else {
      b0 = (q_val >> (i - 1)) & 1;
    }
    b1 = (q_val >> i) & 1;

    if (b0 == b1)
      continue;

    if (b1 == 0) {
      terms[*num_terms] = (-sign) * (1 << i);
    } else {
      terms[*num_terms] = sign * (1 << i);
    }
    (*num_terms)++;
  }
}

template <typename scalar_t>
__global__ void single_cuda_kernel(const scalar_t *__restrict__ input,
                                   scalar_t *__restrict__ output,
                                   const float sf, const int32_t num_keep_terms,
                                   const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int32_t num_terms;
    int32_t terms[MAX_BOOTH_SIZE];
    int32_t sign = input[idx] < 0 ? -1 : 1;
    booth_encode(input[idx], terms, &num_terms, sf);
    output[idx] = 0;
    for (int i = 0; i < num_keep_terms; i++) {
      if (terms[i] == 0) {
        break;
      }
      output[idx] += terms[i];
    }
    output[idx] *= sf;
    output[idx] *= -sign;
  }
}

template <typename scalar_t>
__global__ void group_cuda_kernel(const scalar_t *__restrict__ input,
                                  scalar_t *__restrict__ output, const float sf,
                                  const int32_t group_size,
                                  const int32_t num_keep_terms, const int32_t B,
                                  const int32_t C, const int32_t W,
                                  const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t CWH = C * W * H;
  const int32_t WH = W * H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b * CWH) / WH;
  const int32_t w = (idx - b * CWH - c * WH) / W;
  const int32_t h = idx - b * CWH - c * WH - w * H;
  const int32_t base_offset = b * CWH + w * W + h;
  int32_t gidx;

  if (c < (C / group_size)) {
    int32_t term_idx[MAX_GROUP_SIZE];
    int32_t num_terms[MAX_GROUP_SIZE];
    int32_t terms[MAX_GROUP_SIZE * MAX_BOOTH_SIZE];
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      output[gidx] = 0;
      term_idx[i] = 0;
      booth_encode(input[gidx], &terms[i * MAX_BOOTH_SIZE], &num_terms[i], sf);
    }

    for (int i = 0; i < num_keep_terms; ++i) {
      int32_t max_idx = 0;
      int32_t max_val = 0;
      // loop through groups and add max term
      for (int j = 0; j < group_size; ++j) {
        // find maximum term (of sorted choices)
        int32_t term = terms[j * MAX_BOOTH_SIZE + term_idx[j]];
        if (abs(term) > abs(max_val)) {
          max_val = term;
          max_idx = j;
        }
      }

      // no more useful terms left -- exit
      if (max_val == 0) {
        break;
      }

      // add the max term to correct output
      gidx = (c * group_size + max_idx) * WH + base_offset;
      output[gidx] += max_val;

      // increment pointer to max term element
      term_idx[max_idx]++;
    }

    // multiply each entry by scaling factor
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      int32_t sign = input[gidx] < 0 ? -1 : 1;
      output[gidx] *= sf;
      output[gidx] *= -sign;
    }
  }
}

template <typename scalar_t>
__global__ void top_group_cuda_kernel(const scalar_t *__restrict__ input,
                                      scalar_t *__restrict__ output, const float sf,
                                      const int32_t group_size, const int32_t num_values,
                                      const int32_t num_keep_terms, const int32_t B,
                                      const int32_t C, const int32_t W,
                                      const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t CWH = C * W * H;
  const int32_t WH = W * H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b * CWH) / WH;
  const int32_t w = (idx - b * CWH - c * WH) / W;
  const int32_t h = idx - b * CWH - c * WH - w * H;
  const int32_t base_offset = b * CWH + w * W + h;
  int32_t gidx, hidx;

  if (c < (C / group_size)) {
    // created sorted indexes in descending order
    int32_t sorted_indexes[MAX_VALUES];

    // initialize array
    for (int i = 0; i < group_size; i++) {
      sorted_indexes[i] = i;
    }
    for (int i = 0; i < group_size - 1; i++) {
      for (int j = 0; j < group_size - i - 1; j++) {
        gidx = (c * group_size + sorted_indexes[j]) * WH + base_offset;
        scalar_t value = input[gidx];
        hidx = (c * group_size + sorted_indexes[j+1]) * WH + base_offset;
        scalar_t other_value = input[hidx];

        if (abs(value) < abs(other_value)) {
          int32_t swap = sorted_indexes[j];
          sorted_indexes[j] = sorted_indexes[j+1];
          sorted_indexes[j+1] = swap;
        }
      }
    }
    // end create sorted indexes

    int32_t num_terms;
    int32_t terms[MAX_BOOTH_SIZE];
    for (int i = 0; i < num_values; i++) {
      gidx = (c * group_size + sorted_indexes[i]) * WH + base_offset;
      booth_encode(input[gidx], terms, &num_terms, sf);
      for (int j = 0; j < num_keep_terms; j++) {
        output[gidx] += terms[j];
      }
    }

    // multiply each entry by scaling factor
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      int32_t sign = input[gidx] < 0 ? -1 : 1;
      output[gidx] *= sf;
      output[gidx] *= -sign;
    }
  }
}

template <typename scalar_t>
__global__ void weight_single_cuda_kernel(
    const scalar_t *__restrict__ input, const int32_t *__restrict__ terms,
    scalar_t *__restrict__ output, const float sf, const int32_t total_values,
    const int32_t max_num_terms, const int32_t num_keep_terms,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int32_t q_val = int32_t(input[idx] / sf) + (total_values / 2);
    output[idx] = 0;
    for (int i = 0; i < num_keep_terms; ++i) {
      int32_t term = terms[q_val * max_num_terms + i];

      // a 0 signifies end of encoding
      if (term == 0)
        break;

      int32_t sign = term < 0 ? -1 : 1;
      term = sign * (abs(term) - 1);
      output[idx] += sign * (1 << abs(term));
    }
    output[idx] *= sf;
  }
}

template <typename scalar_t>
__global__ void weight_group_cuda_kernel(
    const scalar_t *__restrict__ input, const int32_t *__restrict__ terms,
    scalar_t *__restrict__ output, const float sf, const int32_t group_size,
    const int32_t total_values, const int32_t max_num_terms,
    const int32_t num_keep_terms, const int32_t B, const int32_t C,
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
    int32_t q_vals[MAX_GROUP_SIZE];
    int32_t term_idx[MAX_GROUP_SIZE];
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      output[gidx] = 0;
      q_vals[i] = int32_t(input[gidx] / sf) + (total_values / 2);
      term_idx[i] = 0;
    }

    for (int i = 0; i < num_keep_terms; ++i) {
      int32_t max_idx = 0;
      int32_t max_val = 0;
      // loop through groups and add max term
      for (int j = 0; j < group_size; ++j) {
        // find maximum term (of sorted choices)
        int32_t term = terms[q_vals[j] * max_num_terms + term_idx[j]];
        if (abs(term) > abs(max_val)) {
          max_val = term;
          max_idx = j;
        }
      }

      // no more useful terms left -- exit
      if (max_val == 0) {
        break;
      }

      // add the max term to correct output
      gidx = (c * group_size + max_idx) * WH + base_offset;
      int32_t sign = max_val < 0 ? -1 : 1;
      output[gidx] += sign * (1 << (abs(max_val) - 1));

      // increment pointer to max term element
      term_idx[max_idx]++;
    }

    // multiply each entry by scaling factor
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      output[gidx] *= sf;
    }
  }
}
} // namespace

at::Tensor single_cuda(const at::Tensor input, const float sf,
                       const int32_t num_keep_terms) {
  const auto size = input.numel();
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "single_cuda", ([&] {
        single_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), output.data<scalar_t>(), sf, num_keep_terms,
            size);
      }));

  return output;
}

at::Tensor group_cuda(const at::Tensor input, const float sf,
                      const int32_t group_size, const int32_t num_keep_terms) {
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

  AT_DISPATCH_FLOATING_TYPES(input.type(), "group_cuda", ([&] {
                               group_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                   input.data<scalar_t>(),
                                   output.data<scalar_t>(), sf, group_size,
                                   num_keep_terms, B, C, W, H);
                             }));

  return output;
}

at::Tensor top_group_cuda(const at::Tensor input, const float sf,
                          const int32_t group_size, const int32_t num_values,
                          const int32_t num_keep_terms) {
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

  AT_DISPATCH_FLOATING_TYPES(input.type(), "top_group_cuda", ([&] {
                               top_group_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                   input.data<scalar_t>(),
                                   output.data<scalar_t>(), sf, group_size,
                                   num_values, num_keep_terms, B, C, W, H);
                             }));

  return output;
}



at::Tensor weight_single_cuda(const at::Tensor input, const at::Tensor terms,
                              const float sf, const int32_t num_keep_terms) {
  const auto size = input.numel();
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  const auto total_values = terms.size(0);
  const auto max_num_terms = terms.size(1);
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "single_cuda", ([&] {
        weight_single_cuda_kernel<scalar_t>
            <<<blocks, threads>>>(input.data<scalar_t>(), terms.data<int32_t>(),
                                  output.data<scalar_t>(), sf, total_values,
                                  max_num_terms, num_keep_terms, size);
      }));

  return output;
}

at::Tensor weight_group_cuda(const at::Tensor input, const at::Tensor terms,
                             const float sf, const int32_t group_size,
                             const int32_t num_keep_terms) {
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
  const auto total_values = terms.size(0);
  const auto max_num_terms = terms.size(1);
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(
      input.type(), "weight_group_cuda", ([&] {
        weight_group_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), terms.data<int32_t>(),
            output.data<scalar_t>(), sf, group_size, total_values,
            max_num_terms, num_keep_terms, B, C, W, H);
      }));

  return output;
}