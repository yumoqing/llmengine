#!/share/vllm-0.8.5/bin/python

# pip install accelerate
from appPublic.worker import awaitify
from appPublic.log import debug
from ahserver.serverenv import get_serverenv
from PIL import Image
import torch
from llmengine.base_chat_llm import BaseChatLLM, llm_register
from transformers import AutoModelForCausalLM, AutoTokenizer

class GptossLLM(BaseChatLLM):
	def __init__(self, model_id):
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype="auto",
			device_map="auto"
		)
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
		inputs = self.tokenizer.apply_chat_template(
			messages, 
			add_generation_prompt=True,
			return_tensors="pt",
			return_dict=True
		).to(self.model.device)
		return inputs

llm_register("gpt-oss", GptossLLM)
