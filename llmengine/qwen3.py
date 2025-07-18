#!/share/vllm-0.8.5/bin/python

# pip install accelerate
from appPublic.worker import awaitify
from appPublic.log import debug
from ahserver.serverenv import get_serverenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from llmengine.base_chat_llm import BaseChatLLM, T2TChatLLM, llm_register

class Qwen3LLM(T2TChatLLM):
	def __init__(self, model_id):
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype="auto",
			device_map="auto"
		)
		if torch.backends.mps.is_available():
			device = torch.device("mps")
			self.model = self.model.to(device)
		self.model_id = model_id

	def build_kwargs(self, inputs, streamer):
		generate_kwargs = dict(
			**inputs,
			streamer=streamer,
			max_new_tokens=32768,
			do_sample=True,
			eos_token_id=self.tokenizer.eos_token_id
		)
		return generate_kwargs

	def _messages2inputs(self, messages):
		debug(f'{messages=}')
		text = self.tokenizer.apply_chat_template(
			messages, 
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=True
		)
		return self.tokenizer([text], return_tensors="pt").to(self.model.device)

llm_register("Qwen/Qwen3", Qwen3LLM)

if __name__ == '__main__':
	import sys
	model_path = sys.argv[1]
	q3 = Qwen3LLM(model_path)
	session = {}
	while True:
		print('input prompt')
		p = input()
		if p:
			if p == 'q':
				break;
			for d in q3.stream_generate(session, p):
				print(d)
				"""
				if not d['done']:
					print(d['text'], end='', flush=True)
				else:
					x = {k:v for k,v in d.items() if k != 'text'}
					print(f'\n{x}\n')
				"""
					

