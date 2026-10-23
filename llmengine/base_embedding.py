import torch
import numpy as np

model_pathMap = {
}
def llm_register(model_key, Klass):
	global model_pathMap
	model_pathMap[model_key] = Klass

def get_llm_class(model_path):
	for k,klass in model_pathMap.items():
		if len(model_path.split(k)) > 1:
			return klass
	print(f'{model_pathMap=}')
	return None

class BaseEmbedding:
	
	def use_mps_if_prosible(self):
		if torch.backends.mps.is_available():
			device = torch.device("mps")
			self.model = self.model.to(device)

	def encode(self, input):
		return self.model.encode(input)

	def embeddings(self, input):
		es = self.encode(input)
		data = []
		for i, e in enumerate(es):
			d = {
				"object": "embedding",
				"index": i,
				"embedding": e.tolist() if isinstance(e, np.ndarray) else e
			}
			data.append(d)
		return {
			"object": "list",
			"data": data,
			"model": self.model_name,
			"usage": {
				"prompt_tokens": 0,
				"total_tokens": 0
			}
		}
	
	def similarity(self, qvector, dcovectors):
		s = self.model.similarity([qvector], docvectors)
		return s[0]

