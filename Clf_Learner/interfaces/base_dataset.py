from abc import abstractmethod
from torch import Tensor
from torch.utils.data import Dataset

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..datasets.tools.standardisation_tools import Standardiser

class BaseDataset(Dataset):

    def __init__(self, X:Tensor, y:Tensor, source_file:str, standardise:bool):
        self.filename:str = source_file
        self.strategic_columns: list[int]

        self._standardiser = None
        if standardise:
            from ..datasets.tools.standardisation_tools import get_standardiser
            self._standardiser = get_standardiser(X)

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("All Datasets must have definition for '__len__' implemented")

    @abstractmethod
    def __getitem__(self, index:int) -> Tensor:
        raise NotImplementedError("All Datasets must have definition for '__getitem__' implemented")

    @abstractmethod
    def size(self) -> tuple[tuple[int,int], tuple[int,int]]:
        # Returns the (num_rows, num_columns) for the X tensor and the Y tensor
        raise NotImplementedError("All Datasets must have definition for 'size' implemented")

    @abstractmethod
    def get_all_vals(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("All Datasets must have definition for 'get_all_vals' implemented")

    @abstractmethod
    def get_strategic_columns(self) -> list[int]|None:
        raise NotImplementedError("All Datasets must have definition for 'get_strategic_columns' implemented")
        
    def get_x_dim(self):
        return self.size()[0][1]
    
    def set_standardiser_device(self, device) -> None:
        if self._standardiser:
            self._standardiser.to(device)

    def invert_standardisation(self, X:Tensor) -> Tensor:
        if self._standardiser:
            return self._standardiser.inverse_transform(X)
        else:
            return X

    def get_standardiser(self) -> 'Standardiser|None':
        return self._standardiser