# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer
from llmengine.base_embedding import BaseEmbedding, llm_register

class Qwen3Embedding(BaseEmbedding):
	def __init__(self, model_id, max_length=8096):
		# Load the model
		self.model = SentenceTransformer(model_id)
		# We recommend enabling flash_attention_2 for better acceleration and memory saving,
		# together with setting `padding_side` to "left":
		# model = SentenceTransformer(
		#     "Qwen/Qwen3-Embedding-0.6B",
		#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
		#     tokenizer_kwargs={"padding_side": "left"},
		# )
		self.max_length = max_length
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]

llm_register('Qwen3-Embedding', Qwen3Embedding)
