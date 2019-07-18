import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import torch.nn.functional as F

cgm_cuda = load(
    'cgm_cuda', ['kernels/cgm_cuda.cpp', 'kernels/cgm_cuda_kernel.cu'], extra_cflags=['-O3'])

class StaticCGM(nn.Module):
    def __init__(self, columns):
        super(StaticCGM, self).__init__()
        self.columns = columns

    def forward(self, x):
        B, C, W, H = x.shape
        x = x.permute(1, 0, 2, 3).contiguous().view(C, B*W*H)
        x_out = torch.zeros_like(x)
        for columns in self.columns:
            xi = torch.index_select(x, 0, torch.Tensor(columns).long().cuda(0))
            max_idx = torch.max(xi, dim=0)[1]
            idx_rng = torch.arange(B*W*H)

            for i in reversed(range(len(columns))):
                max_idx[max_idx == i] = columns[i]
        
            x_out[max_idx, idx_rng] = x[max_idx, idx_rng]

        x_out = x_out.view(C, B, W, H).permute(1, 0, 2, 3).contiguous()
        return F.relu(x_out)

class CGM(nn.Module):
    def __init__(self, group_size, max_clamp=1e10):
        super(CGM, self).__init__()
        self.group_size = group_size
        self.max_clamp = max_clamp
    
    def forward(self, x):
        return cgm.apply(x, self.group_size, self.max_clamp)

    def extra_repr(self):
        return 'group_size={group_size}, max_clamp={max_clamp}'.format(**self.__dict__)

class cgm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size, max_clamp):
        if x.is_cuda:
            h = cgm_cuda.forward(x, group_size, max_clamp)
        else:
            assert False
            '''
            B, C, W, H = x.shape
            h = x.permute(2, 3, 0, 1).contiguous().view(-1, group_size)
            h[h != h.max(dim=1, keepdim=True)[0]] = 0
            h[h < 0] = 0
            h = h.view(W, H, B, C).permute(2, 3, 0, 1).contiguous()
            '''

        ctx.save_for_backward(h)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        h, = ctx.saved_tensors
        if grad_output.is_cuda:
            grad_output = cgm_cuda.backward(grad_output, h)
        else:
            assert False
            '''
            grad_output[h <= 0] = 0
            '''

        return grad_output, None, None

if __name__=='__main__':
    from torch.autograd import Variable
    import gradcheck
    x = Variable(torch.Tensor(2, 8, 16, 16).uniform_(-1, 1).double().cuda())
    x.requires_grad = True

    for i in range(1, 9):
        layer = CGM(i, 1.0).cuda()
        res = gradcheck.gradcheck(layer, x)
        print('CGM({}): {}'.format(i, res))
