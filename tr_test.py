import torch
import booth


torch.manual_seed(42)
x = torch.Tensor(1, 4, 1, 1).normal_().cuda()
xh = booth.booth_cuda.radix_2_mod(x, 2**-4, 8, 8, 32)
print(xh)