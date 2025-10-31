# m2m/base_m2m.py
import torch
from typing import List, Dict


model_pathMap = {}


def llm_register(model_key, Klass):
    model_pathMap[model_key] = Klass


def get_llm_class(model_path):
    for k, klass in model_pathMap.items():
        if k in model_path:
            return klass
    print(f'{model_pathMap=}')
    return None


class BaseM2M:
    def __init__(self, model_id, **kw):
        self.model_id = model_id

    def m2m(self, texts: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError