import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from llmengine.base_reranker import BaseReranker, llm_register

class Qwen3Reranker(BaseReranker):
	def __init__(self, model_id, max_length=8096):
		if 'Qwen3-Reranker' not in model_id:
			e = Exception(f'{model_id} is not a Qwen3-Reranker')
			raise e
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
		self.model = AutoModelForCausalLM.from_pretrained(model_id).eval()
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]
		self.max_length = 8192

llm_register('Qwen3-Reranker', Qwen3Reranker)
