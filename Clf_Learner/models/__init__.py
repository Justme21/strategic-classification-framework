from ..interfaces import BaseModel
from .linear_model import LinearModel
from .mlp_model import MLPModel
from .parabolic_model import ParabolicModel
from .quadratic_model import QuadraticModel
from .icnn_model import ICNNModel

MODEL_DICT: dict[str, type[BaseModel]] = {
    "linear": LinearModel,
    "mlp": MLPModel,
    "parabolic": ParabolicModel,
    "quadratic": QuadraticModel,
    "icnn": ICNNModel,
}


