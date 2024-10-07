from pyapprox.sciml.util._utils import BaseUtilitiesSciML
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin

class TorchUtilitiesSciML(BaseUtilitiesSciML, TorchLinAlgMixin):
    def __init__(self):
        super().__init__()
