import torch
import torch.nn as nn

from ..interfaces import BaseBestResponse, BaseDataset, BaseLoss, BaseModel
from .tools.model_training_tools import vanilla_training_loop

class LinearModel(BaseModel, nn.Module):
    """Standard linear model"""
    def __init__(self, best_response:BaseBestResponse, loss:BaseLoss, address:str, x_dim:int, **kwargs):
        BaseModel.__init__(self, best_response, loss, address, x_dim)
        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.weight = nn.Parameter(torch.randn(1, x_dim))
        self.bias = nn.Parameter(torch.randn(1))

        self.best_response = best_response
        self.loss = loss

    def get_boundary_vals(self, X):
        """(Optional) For the input 1-D X values, returns the y values that would lie
            on the model decision boundary. This is only used for data visualisation (not included in repo)"""
        W = self.weight[0]
        b = self.bias

        y = (-W[0]*X-b)*(1.0/W[1]) 
        boundary_coords = torch.stack([X,y], dim=1)
        return boundary_coords

    def get_weights(self, include_bias=True) -> torch.Tensor:
        weights = self.weight
        if include_bias:
            bias = self.bias.unsqueeze(0)
            weights = torch.cat((weights, bias), dim=1)
        
        return weights
    
    def forward(self, X):
        # Flatten to make output uni-dimensional to match y
        out = torch.einsum("bi, oi->bo", X, self.weight) + self.bias
        return out.squeeze()

    def predict(self, X):
        y_hat = self.forward(X)
        y_hat[torch.abs(y_hat) <= 1e-10] = 0 # This is a dangerous stopgap, we later map negatives to 0.
        return torch.sign(y_hat)

    def fit(self, train_dset:BaseDataset, opt, lr:float, batch_size:int=128, epochs:int=100, validate:bool=False, verbose:bool=False) -> dict:
        train_losses_dict = vanilla_training_loop(self, train_dset, opt, lr, batch_size, epochs, validate, verbose)
        return train_losses_dict