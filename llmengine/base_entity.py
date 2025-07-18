from abc import ABC, abstractmethod
from typing import List

model_pathMap = {}

def ltp_register(model_key, Klass):
    """Register a model class for a given model key."""
    global model_pathMap
    model_pathMap[model_key] = Klass

def get_ltp_class(model_path):
    """Find the model class for a given model path."""
    for k, klass in model_pathMap.items():
        if len(model_path.split(k)) > 1:
            return klass
    print(f'{model_pathMap=}')
    return None

class BaseLtp(ABC):
    @abstractmethod
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query text."""
        pass