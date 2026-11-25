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
		if getattr(self, 'model', None) is None:
			# OpenVINO 等非 torch 后端没有 self.model，直接返回
			return
		if torch.backends.mps.is_available():
			device = torch.device("mps")
			self.model = self.model.to(device)

	def encode(self, input):
		return self.model.encode(input)

	def embeddings(self, input):
		r = self.encode(input)
		# encode() 可以返回 (vectors, prompt_tokens)，也可以只返回 vectors（旧后端）
		if isinstance(r, tuple) and len(r) == 2:
			es, prompt_tokens = r
		else:
			es, prompt_tokens = r, 0
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
			# 真实 token 数作为计费事实源。以前恒写 0，挂 ppid 也永远出账 0 元。
			"usage": {
				"prompt_tokens": int(prompt_tokens),
				"total_tokens": int(prompt_tokens)
			}
		}
	
	def similarity(self, qvector, dcovectors):
		s = self.model.similarity([qvector], docvectors)
		return s[0]