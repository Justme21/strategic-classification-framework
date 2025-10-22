from ..interfaces import BaseBestResponse

from .augmented_lagrangian_best_response import AugmentedLagrangianBestResponse
from .identity_best_response import IdentityResponse
from .lagrangian_best_response import LagrangianBestResponse
from .linear_best_response import LinearBestResponse
from .gradient_ascent_best_response import GradientAscentBestResponse

BR_DICT: dict[str, type[BaseBestResponse]] = {
    "augmented_lagrange": AugmentedLagrangianBestResponse,
    "gradient": GradientAscentBestResponse,
    "identity": IdentityResponse,
    "lagrange": LagrangianBestResponse,
    "linear": LinearBestResponse,
}