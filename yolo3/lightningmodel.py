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
class Model(L.LightningModule):
    def __init__(self, optimizer,scheduler):
        super().__init__()
        ## call the encoder/decoder etc
        # self.model = 
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        

    
        

    def forward(self, x): #prediction/inference actions
        pass
    
    def training_step(self, batch, batch_idx): #training loop independent of forward
        
        loss, total_location_loss, total_confidence_loss, total_class_loss, total_dist_loss = yolo3_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, elim_grid_sense=False, use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=True
        return
    
    
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return self.optimizer

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
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.layer1 = nn.LazyLinear(32)
#         self.layer2 = nn.LazyLinear(5)

#     def call(self, inputs):
#         x = nn.ReLU(self.layer1(inputs))
#         x = nn.Softmax(self.layer2(x), dim=1)
#         return self.layer2(x)