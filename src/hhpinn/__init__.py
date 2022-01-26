# flake8: noqa
__version__ = "0.2"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

from . import datasets
from . import models
from . import plotting

from .models import AveragingModel
from .models import StreamFunctionPINN, HHPINN2D, SequentialHHPINN2D
