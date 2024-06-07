import torch
import pytorch_lightning as L
from torch import nn
from torch import Tensor

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class MyModel(L.LightningModule):
    def __init__(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

class Input(nn.Module):
    def __init__(self, shape:tuple, device=torch.device('cpu'), name="input_type", dtype=torch.float32):
        super().__init__()
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.name = name

    def forward(self, x):
        tensor_shape = tuple(x.shape[1:])

        if x.device != self.device:
            raise ValueError(f"Input tensor must be on device {self.device} but got device {x.device} instead")
        if x.dtype != self.dtype:
            raise ValueError(f"Input tensor must have data type {self.dtype} but got data type {x.dtype} instead")
        if tensor_shape != self.shape:
            raise ValueError(f"Input shape must be {self.shape} but got {tensor_shape} instead")
        
        return (x.to(self.device, self.dtype, self.name))
