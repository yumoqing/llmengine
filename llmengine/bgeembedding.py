import os

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from llmengine.base_embedding import BaseEmbedding, llm_register


def _backend_choice():
	return (os.environ.get('LLMENGINE_EMBED_BACKEND') or 'auto').lower()


class BgeEmbedding(BaseEmbedding):
	"""bge 系列向量化。

	后端可插拔（LLMENGINE_EMBED_BACKEND = auto | ov | torch）：
	- ov/auto：OpenVINO（Intel CPU / iGPU）。设备由 LLMENGINE_EMBED_DEVICE = AUTO|GPU|CPU 决定。
	  2026-11 nuc115 实测（bge-m3）：32 条短文本 4202 ms → 314 ms，903 token 长文本 2981 ms → 589 ms。
	- torch：老路径，保留作退路；已修成一次批处理（原来逐条 embed_query，等于没有批处理）。

	encode() 返回 (vectors, prompt_tokens)，prompt_tokens 为截断后的真实 token 数 —— 修掉
	「usage 恒 0 导致 embedding 调用出账 0 元」的旧缺陷。基类同时兼容旧的「只返 vectors」写法。
	"""

	def __init__(self, model_id, max_length=None):
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]
		self.model = None
		self.ov = None
		self.max_length = max_length
		choice = _backend_choice()
		if choice in ('auto', 'ov', 'openvino'):
			try:
				from llmengine.ov_runtime import OvEncoder
				self.ov = OvEncoder(model_id, max_length=max_length, strict=(choice != 'auto'))
				print(f'[bgeembedding] backend=openvino device={self.ov.device_name} '
					  f'max_length={self.ov.max_length}')
			except Exception as e:
				if choice != 'auto':
					raise
				print(f'[bgeembedding] openvino unavailable ({e}), fallback to torch')
		if self.ov is None:
			self.model = HuggingFaceEmbeddings(
				model_name=model_id,
				model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
				encode_kwargs={
					"batch_size": 12,
					'normalize_embeddings': True
				}
			)

	def encode(self, input):
		if self.ov is not None:
			return self.ov.encode(input)
		texts = [t if isinstance(t, str) else str(t) for t in input]
		if not texts:
			return [], 0
		vectors = self.model.embed_documents(texts)
		return vectors, self.count_tokens(texts)

	def count_tokens(self, texts):
		if self.ov is not None:
			return sum(len(x) for x in self.ov.tokenizer(
				texts, truncation=True, max_length=self.ov.max_length, padding=False)['input_ids'])
		try:
			return sum(len(x['input_ids']) for x in self.model.client.tokenize(texts))
		except Exception:
			return 0


llm_register('bge-m3', BgeEmbedding)