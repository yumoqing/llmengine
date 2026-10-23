import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llmengine.base_reranker import BaseReranker, llm_register
from torch.cuda.amp import autocast

class BgeReranker(BaseReranker):
    def __init__(self, model_id, max_length=8096):
        if 'bge-reranker' not in model_id:
            e = Exception(f'{model_id} is not a bge-reranker')
            raise e
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            model = model.to('cuda')
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        self.model = model
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]

    def build_pairs(self, query, docs, **kw):
        return [[query, doc] for doc in docs]

    def process_inputs(self, pairs):
        inputs = self.tokenizer(pairs, padding=True,
                        truncation=True, return_tensors='pt', max_length=8096)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        return inputs

    def compute_logits(self, inputs):
        if torch.cuda.is_available():
            with autocast():
                scores = self.model(**inputs, return_dict=True).logits.view(-1,)
        else:
            with torch.no_grad():
                scores = self.model(**inputs, return_dict=True).logits.view(-1,)
        scores = [s.item() for s in scores]
        return scores

llm_register('bge-reranker', BgeReranker)