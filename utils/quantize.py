"""utils function for quantization"""
import torch

class Function_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        """
        This function quantizes normalized and unbiased input x (within [-1, 1]) to discrete values (within {-1, 1}).
        It can only process multi-bit quantization, binarization reduce to [-1, 0]
        Args:
            ctx:
            x:
            bit:

        Returns:

        """
        ctx.save_for_backward(x)

        n = float(2 ** (bit-1))
        _round_x = torch.round(x * n)  # [-1, 1] => [n, n]
        # {-n, n, 1} => {-n, n-1, 1}
        _quantized_bit = torch.clip(_round_x, -n, n-1)
        return _quantized_bit / n, _quantized_bit


    @staticmethod
    def backward(ctx, grad_outputs, _):
        x, = ctx.saved_tensors
        gate = (torch.abs(x) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


def vector_wise_absmax_quantize(_x, _bit, _vector_index):
    """
    This function quantizes input _x into _bit, with the elements in _vector_index share a same scaling factor
    Args:
        _x:
        _bit:
        _vector_index:

    Returns:

    """
    if _bit == 32:
        return _x, None

    left =  torch.amax(_x, dim=_vector_index, keepdim=True)
    right = torch.amin(_x, dim=_vector_index, keepdim=True)
    mid = (right + left) / 2
    _shift_x = _x - mid # [left, right] => [left-mid, right-mid]
    _scaling_factor = torch.amax(torch.abs(_shift_x), dim=_vector_index, keepdim=True) + 1e-6
    scale_x = _shift_x / _scaling_factor # [left-mid, right-mid] => [-1, 1]
    _scale_quantized_x, _quantized_bit = Function_STE.apply(scale_x, _bit) # [-1, 1] => {-1, 1}
    _quantized_x = _scale_quantized_x * _scaling_factor + mid
    # return _quantized_x, _quantized_bit # , _scaling_factor, mid
    return _quantized_x, None