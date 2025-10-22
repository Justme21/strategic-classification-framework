from ..interfaces import BaseUtility

from .hinge_utility import HingeUtility
from .strategic_utility import StrategicUtility
from .strategic_sigmoid_utility import StrategicSigmoidUtility
from .strategic_tanh_utility import StrategicTanhUtility

UTILITY_DICT: dict[str, type[BaseUtility]]= {
    "hinge": HingeUtility,
    "strategic": StrategicUtility,
    "strategic_sigmoid": StrategicSigmoidUtility,
    "strategic_tanh": StrategicTanhUtility,
}