# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer
from llmengine.base_embedding import BaseEmbedding, llm_register

class Qwen3Embedding(BaseEmbedding):
	def __init__(self, model_id, max_length=8096):
		import torch
		# Use float32 on CPU to avoid bfloat16 inference issues
		if torch.cuda.is_available():
			self.model = SentenceTransformer(model_id)
		else:
			self.model = SentenceTransformer(model_id, model_kwargs={"torch_dtype": torch.float32})
		self.max_length = max_length
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]

llm_register('Qwen3-Embedding', Qwen3Embedding)
