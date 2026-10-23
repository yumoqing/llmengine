
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from llmengine.base_embedding import BaseEmbedding, llm_register

class BgeEmbedding(BaseEmbedding):
	def __init__(self, model_id):
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]
		self.model = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={
				"batch_size": 12,
				'normalize_embeddings': True
			}
        )

	def encode(self, input):
		ret = []
		for t in input:	
			embedding = self.model.embed_query(t)
			ret.append(embedding)
		return ret

llm_register('bge-m3', BgeEmbedding)
