import torch

from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base_best_response import BaseBestResponse
    from .base_dataset import BaseDataset
    from .base_loss import BaseLoss

class BaseModel(ABC):
    def __init__(self, best_response:'BaseBestResponse', loss:'BaseLoss', address:str, x_dim:int|None=None):
        # These are defined here so that the type-hinting is consistent
        self.address: str = address
        self.best_response: BaseBestResponse
        self.loss: BaseLoss
        self.x_dim: int

    def get_num_components(self) -> int:
        return 1

    def get_mixture_probs(self) -> Tensor:
        return torch.tensor(1.0)

    # Adding forward variants to handle the case where the forward function called in the best response (or the loss) might not be the standard forward
    def forward_utility(self, X:Tensor, i:int|None=None) -> Tensor:
        """Version of the forward function that is called in the utility definition. By default this is just forward"""
        return self.forward(X)
    
    def forward_loss(self, X:Tensor) -> Tensor:
        """Version of the forward function that is called in the loss definition. By default this is just forward"""
        return self.forward(X)
    
    def get_boundary_vals(self, X:Tensor) -> Tensor|list:
        """(Optional) For the input 1-D X values, returns the y values that would lie
            on the model decision boundary. This is only used for data visualisation (not included in repo)"""
        raise NotImplementedError

    def save_params(self, address:str|None=None) -> None:
        """ Save model parameters to a file
        : address (str): address to save model parameters to
        : return: None
        """
        assert isinstance(self, torch.nn.Module), "Error: For non torch-based models save_params and load_params functions have to be implemented"
        if address is None:
            address = self.address
        torch.save(self.state_dict(), f"{address}/model_params")
 
    def load_params(self, address:str|None=None) -> None:
        """ Load model parameters from a file
        : address (str) address of the file with the model parameters
        : return: None
        """
        assert isinstance(self, torch.nn.Module), "Error: For non torch-based models save_params and load_params functions have to be implemented"
        if address is None:
            address = self.address
        #self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True))
        # For loading weights from model trained on Cuda device
        self.load_state_dict(torch.load(f"{address}/model_params", weights_only=True, map_location=torch.device('cpu')))
        
    def to(self, device:str) -> None:
        """ Load model weights onto device
        : device (str) identifier for device to load models weights to
        : return: None"""
        for w in self.get_weights():
            w.to(device)

    @abstractmethod
    def get_weights(self, include_bias:bool=True) -> Tensor:
        """Return the model weights
        : return: model weights
        """
        pass

    @abstractmethod
    def fit(self, train_dset:'BaseDataset', opt, lr:float, batch_size:int, epochs:int, val_dset:'BaseDataset', validate:bool, verbose:bool) -> dict:
        """ Learn to predict the true y values associated with the given Xs
        : X: Data to learn from
        : y: True values
        : return: dict of containing training metrics
        """
        pass

    @abstractmethod
    def forward(self, X:Tensor) -> Tensor:
        """ Evaluate the output associated with input X
        : X: Data to be evaluated
        : return: Model Prediction
        """
        pass

    @abstractmethod
    def predict(self, X:Tensor) -> Tensor:
        """ Predict the y for the given X
        : X: Data to predict
        : return: Model Prediction
        """
        # This will often just be a call to forward, but formatted to be output friendly
        pass

    #@abstractmethod
    #def parameters(self) -> Tensor:
    #    # Mocking the pytorch model parameters function. This function is specified automatically for pytorch models
    #    pass