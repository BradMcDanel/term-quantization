#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
at::Tensor binary_cuda(const at::Tensor input,  const float sf,
                       const int32_t group_size, const int32_t num_keep_terms);

at::Tensor radix_2_mod_cuda(const at::Tensor input,  const float sf,
                            const int32_t group_size, const int32_t num_keep_terms);

at::Tensor value_group_cuda(const at::Tensor input, const int32_t group_size,
                            const int32_t num_keep_values);

at::Tensor single_cuda(const at::Tensor input, const float sf,
                       const int32_t num_keep_terms);

at::Tensor weight_single_cuda(const at::Tensor input, const at::Tensor terms,
                         const float sf, const int32_t num_keep_terms);

at::Tensor weight_group_cuda(const at::Tensor input, const at::Tensor terms, const float sf,
                             const int32_t group_size, const int32_t num_keep_terms);

at::Tensor group_cuda(const at::Tensor input, const float sf,
                      const int32_t group_size, const int32_t num_keep_terms);

at::Tensor top_group_cuda(const at::Tensor input, const float sf,
                          const int32_t group_size, const int32_t num_values,
                          const int32_t num_keep_terms);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

at::Tensor binary(const at::Tensor input, const float sf,
                  const int32_t group_size, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return binary_cuda(input, sf, group_size, num_keep_terms);
}

at::Tensor radix_2_mod(const at::Tensor input, const float sf,
                       const int32_t group_size, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return radix_2_mod_cuda(input, sf, group_size, num_keep_terms);
}

at::Tensor value_group(const at::Tensor input, const int32_t group_size,
                       const int32_t num_keep_values) {
  CHECK_INPUT(input);
  return value_group_cuda(input, group_size, num_keep_values);
}

at::Tensor single(const at::Tensor input, const float sf,
                  const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return single_cuda(input, sf, num_keep_terms);
}

at::Tensor group(const at::Tensor input, const float sf,
                 const int32_t group_size, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return group_cuda(input, sf, group_size, num_keep_terms);
}

at::Tensor top_group(const at::Tensor input, const float sf,
                     const int32_t group_size, const int32_t num_values,
                     const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return top_group_cuda(input, sf, group_size, num_values, num_keep_terms);
}

at::Tensor weight_single(const at::Tensor input, const at::Tensor terms,
                         const float sf, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  CHECK_INPUT(terms);
  return weight_single_cuda(input, terms, sf, num_keep_terms);
}

at::Tensor weight_group(const at::Tensor input, const at::Tensor terms, const float sf,
                        const int32_t group_size, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  CHECK_INPUT(terms);
  return weight_group_cuda(input, terms, sf, group_size, num_keep_terms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary", &binary, "Binary Term Revealing (CUDA)");
  m.def("radix_2_mod", &radix_2_mod, "Booth (Modified Radix-2) (CUDA)");
  m.def("value_group", &value_group, "Value Grouping (CUDA)");
  m.def("single", &single, "Booth single (CUDA)");
  m.def("group", &group, "Booth group (CUDA)");
  m.def("top_group", &top_group, "Top-k values Booth group (CUDA)");
  m.def("weight_single", &weight_single, "Weight single (CUDA)");
  m.def("weight_group", &weight_group, "Weight group (CUDA)");
}
