#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor cgm_cuda_forward(
    const at::Tensor input,
    const int32_t group_size, 
    const float max_clamp);

at::Tensor cgm_cuda_backward(
    at::Tensor grad_input,
    const at::Tensor input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor cgm_forward(
    const at::Tensor input,
    const int32_t group_size,
    const float max_clamp) {
  CHECK_INPUT(input);

  return cgm_cuda_forward(input, group_size, max_clamp);
}

at::Tensor cgm_backward(
    at::Tensor grad_input,
    const at::Tensor input) {
  CHECK_INPUT(grad_input);
  CHECK_INPUT(input);
  return cgm_cuda_backward(grad_input, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cgm_forward, "CGM forward (CUDA)");
  m.def("backward", &cgm_backward, "CGM backward (CUDA)");
}
