import torch
from abc import ABC, abstractmethod
from appPublic.log import debug
import os

model_pathMap = {}

def llm_register(model_key: str, Klass):
    """Register a triplet extractor class for a given model key."""
    global model_pathMap
    model_pathMap[model_key] = Klass
    debug(f"Registered {Klass.__name__} for model_key: {model_key}")


def get_llm_class(model_path: str):
    """Return the triplet extractor class for the given model path."""
    for k, klass in model_pathMap.items():
        if k in model_path:
            return klass
    debug(f"No class found for model_path: {model_path}, model_pathMap: {model_pathMap}")
    return None


class BaseTripleExtractor(ABC):
    """Base class for triplet extraction."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.model = None
        debug(f"Initialized BaseTripleExtractor with model_path: {model_path}")

    def use_mps_if_possible(self):
        """Select device (MPS, CUDA, or CPU)."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            debug("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            debug("Using CUDA device")
        else:
            device = torch.device("cpu")
            debug("Using CPU device")
        if self.model is not None:
            self.model = self.model.to(device)
        return device

    @abstractmethod
    async def extract_triplets(self, text: str) -> list:
        """Extract triplets from text."""
        pass
