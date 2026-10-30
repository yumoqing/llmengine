import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llmengine.base_reranker import BaseReranker, llm_register
from torch.cuda.amp import autocast

class BgeReranker(BaseReranker):
	def __init__(self, model_id, max_length=None):
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
		# 2026-10-30 500 事故：原来硬编码 max_length=8096 做 truncation，
		# 而 bge-reranker-base 的 max_position_embeddings 只有 514 —— 长输入根本
		# 不会被截断，直接进 forward 越界抛 IndexError，接口回 HTTP 500。
		# 取模型真实上限（tokenizer.model_max_length / config.max_position_embeddings），
		# 显式传参才允许覆盖（如 v2-m3 的 8194）。
		self.max_length = self._resolve_max_length(max_length)

	def _resolve_max_length(self, max_length):
		limits = []
		if max_length:
			limits.append(int(max_length))
		tk_max = getattr(self.tokenizer, 'model_max_length', None)
		if tk_max and tk_max < 10 ** 6:  # 未设置时 transformers 给一个超大哨兵值
			limits.append(int(tk_max))
		cfg_max = getattr(getattr(self.model, 'config', None), 'max_position_embeddings', None)
		if cfg_max:
			limits.append(int(cfg_max))
		if not limits:
			return 512
		return min(limits)

	def build_pairs(self, query, docs, **kw):
		return [[query, doc] for doc in docs]

	def process_inputs(self, pairs):
		# truncation='longest_first'：query/document 谁长截谁，保证不超过模型上限，
		# 既不 500，也不会把 query 整条截没（默认 'longest_first' 已是最优，显式声明）。
		inputs = self.tokenizer(pairs, padding=True, truncation='longest_first',
						return_tensors='pt', max_length=self.max_length)
		if torch.cuda.is_available():
			inputs = {k: v.to('cuda') for k, v in inputs.items()}
		return inputs

	def compute_logits(self, inputs, **kwargs):
		if torch.cuda.is_available():
			with autocast():
				scores = self.model(**inputs, return_dict=True).logits.view(-1,)
		else:
			with torch.no_grad():
				scores = self.model(**inputs, return_dict=True).logits.view(-1,)
		scores = [s.item() for s in scores]
		return scores

llm_register('bge-reranker', BgeReranker)