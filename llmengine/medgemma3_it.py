# pip install accelerate
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
from llmengine.base_chat_llm import BaseChatLLM, llm_register

model_id = "google/medgemma-4b-it"

class MedgemmaLLM(BaseChatLLM):
	def __init__(self, model_id):
		self.model = AutoModelForImageTextToText.from_pretrained(
			model_id,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)
		self.processor = AutoProcessor.from_pretrained(model_id)
		self.tokenizer = self.processor.tokenizer
		self.model_id = model_id

	def _messages2inputs(self, messages):
		inputs = self.processor.apply_chat_template(
			messages, 
			add_generation_prompt=True, 
			tokenize=True,
			return_dict=True, 
			return_tensors="pt"
		).to(self.model.device, dtype=torch.bfloat16)
		return inputs

llm_register("google/medgemma", MedgemmaLLM)

if __name__ == '__main__':
	med = MedgemmaLLM('/share/models/google/medgemma-4b-it')
	session = {}
	while True:
		print(f'chat with {med.model_id}')
		print('input prompt')
		p = input()
		if p:
			if p == 'q':
				break;
			print('input image path')
			imgpath=input()
			for d in med.stream_generate(session, p, image_path=imgpath):
				if not d['done']:
					print(d['text'], end='', flush=True)
				else:
					x = {k:v for k,v in d.items() if k != 'text'}
					print(f'\n{x}\n')
					

