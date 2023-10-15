import timm
import torch
from torch import nn

import incept_next_quantization.inception_next.inceptionnext


class QuantModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)
