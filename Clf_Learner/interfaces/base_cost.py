from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..datasets.tools.standardisation_tools import Standardiser

class BaseCost(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self._standardiser = None

    @abstractmethod
    def __call__(self, X:Tensor, Z:Tensor) -> Tensor:
        pass

    def set_standardiser(self, standardiser:'Standardiser|None') -> None:
        self._standardiser = standardiser

    def get_standardiser(self) -> 'Standardiser|None':
        return self._standardiser