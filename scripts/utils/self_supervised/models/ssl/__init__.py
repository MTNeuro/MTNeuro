from .byol import BYOL
from .double_byol import DoubleBYOL
from .myow_factory import myow_factory
from .simclr import SimCLR

# generate myow variants
MYOW = myow_factory(DoubleBYOL)

__all__ = [
    'BYOL',
    'MYOW',
    'SimCLR'
]
