import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llmengine.base_reranker import BaseReranker, llm_register

class BgeReranker(BaseReranker):
	def __init__(self, model_id, max_length=8096):
		if 'bge-reranker' not in model_id:
			e = Exception(f'{model_id} is not a bge-reranker')
			raise e
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		model = AutoModelForSequenceClassification.from_pretrained(model_id)
		model.eval()
		self.model = model
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]

	def build_pairs(self, query, docs, **kw):
		return [[query, doc] for doc in docs]

	def process_inputs(self, pairs):
		inputs = self.tokenizer(pairs, padding=True, 
						truncation=True, return_tensors='pt', max_length=512)
		return inputs

	def compute_logits(self, inputs):
		scores = self.model(**inputs, 
						return_dict=True).logits.view(-1, ).float()
		scores = [ s.item() for s in scores ]
		return scores

llm_register('bge-reranker', BgeReranker)
