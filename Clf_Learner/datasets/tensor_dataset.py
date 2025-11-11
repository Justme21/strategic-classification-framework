from torch import Tensor
from torch.utils.data import Dataset

from .tools.standardisation_tools import get_standardiser

from ..interfaces import BaseDataset

class TensorDataset(BaseDataset, Dataset):
    # TODO: Ape the pytorch TensorDataset
    # train_dset = TensorDataset(X, r, y)

    def __init__(self, X:Tensor, y:Tensor, filename="", standardise=True, **kwargs):
        super().__init__(X, y, filename, standardise)
        assert len(y.shape) == 1, f"Error: downstream models expect target tensor to have a single dimension, current target tensor has shape {y.shape}"
        
        if standardise:
            assert self._standardiser is not None # If standardise is True then the standardiser should be set
            print("Standardising Dataset")
            X = self._standardiser.transform(X)

        self.X = X
        self.y = y

        self.filename = filename

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def size(self):
        return self.X.size(), self.y.size()
    
    def get_all_vals(self) -> tuple[Tensor, Tensor]:
        return self.X, self.y