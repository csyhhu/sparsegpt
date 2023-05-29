"""utils function for quantized module"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantize import vector_wise_absmax_quantize

class quantized_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias, bitW=8, bitA=8, vector_index=[0, 1]):
        super(quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.bitW = bitW
        self.bitA = bitA
        self.vector_index = vector_index

        self.quantized_input = None
        self.quantized_input_bit = None
        self.quantized_weight = None
        self.quantized_weight_bit = None

        print('Initialized Quantized Linear with bitW: {}, bitA: {}, vector index: {}'.format(self.bitW, self.bitA, self.vector_index))

    def forward(self, input):

        self.quantized_input, self.quantized_input_bit = vector_wise_absmax_quantize(input, self.bitA, _vector_index=self.vector_index)
        self.quantized_weight, self.quantized_weight_bit = vector_wise_absmax_quantize(self.weight, self.bitW, _vector_index=self.vector_index)

        output = F.linear(self.quantized_input, self.quantized_weight, self.bias)
        return output


if __name__ == '__main__':

    import torch

    batch_size = 10
    dim1 = 4
    dim2 = 5
    bitW = 8
    bitA = 8
    vector_index = [0, 1]

    inputs = 2 * (torch.rand([batch_size, dim1]) - 0.5)
    labels = 2 * (torch.rand([batch_size, dim2]) - 0.5)
    layer = quantized_Linear(in_features=dim1, out_features=dim2, bitW=bitW, bitA=bitA, vector_index=vector_index)
    outputs = layer(inputs)
    losses = torch.nn.MSELoss()(outputs, labels)
    losses.backward()