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



# hooks for training 
# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightning-hooks
class MyModel(L.LightningModule):
    def __init__(self):
        pass
    
    def forward(self, x): #prediction/inference actions
        pass
    def training_step(self, batch, batch_idx): #training loop independent of forward
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
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

## Attempt to recreate part of "Model()" class. Uncertain of it's usefulness
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.LazyLinear(32)
        self.layer2 = nn.LazyLinear(5)

    def call(self, inputs):
        x = nn.ReLU(self.layer1(inputs))
        x = nn.Softmax(self.layer2(x), dim=1)
        return self.layer2(x)