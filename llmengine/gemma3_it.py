#!/share/vllm-0.8.5/bin/python

# pip install accelerate
import threading
from time import time
from appPublic.worker import awaitify
from ahserver.serverenv import get_serverenv
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from PIL import Image
import requests
import torch
from llmengine.base_chat_llm import BaseChatLLM, llm_register

class Gemma3LLM(BaseChatLLM):
	def __init__(self, model_id):
		self.model = Gemma3ForConditionalGeneration.from_pretrained(
			model_id, device_map="auto"
		).eval()
		self.processor = AutoProcessor.from_pretrained(model_id)
		self.tokenizer = self.processor.tokenizer
		self.messages = []
		self.model_id = model_id

llm_register("gemma-3", Gemma3LLM)

if __name__ == '__main__':
	def get_stream_text(chunk):
		chunk = chunk[6:]
		d = json.loads(chunk)
		return d['choices'][0]['delta']['content']
	gemma3 = Gemma3LLM('/share/models/google/gemma-3-4b-it')
	session = {}
	while True:
		print('input prompt')
		p = input()
		if p:
			if p == 'q':
				break;
			print('input image path')
			imgpath=input()
			for d in gemma3.stream_generate(session, p, image_path=imgpath):
				if not d['DONE']:
					print(d['text'], end='', flush=True)
				else:
					x = {k:v for k,v in d.items() if k != 'text'}
					print(f'\n{x}\n')
					

