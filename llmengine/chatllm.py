from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from time import time
import torch
from threading import Thread

def is_chat_model(model_name: str, tokenizer) -> bool:
	chat_keywords = ["chat", "chatml", "phi", "llama-chat", "mistral-instruct"]
	if any(k in model_name.lower() for k in chat_keywords):
		return True
	if tokenizer and hasattr(tokenizer, "additional_special_tokens"):
		if any(tag in tokenizer.additional_special_tokens for tag in ["<|user|>", "<|system|>", "<|assistant|>"]):
			return True
	return False

def build_chat_prompt(messages):
	prompt = ""
	for message in messages:
		role = message["role"]
		content = message["content"]
		prompt += f"<|{role}|>\n{content}\n"
	prompt += "<|assistant|>\n"  # 生成开始
	return prompt

class CountingStreamer(TextIteratorStreamer):
	def __init__(self, tokenizer, skip_prompt=True, **kw):
		super().__init__(tokenizer, skip_prompt=skip_prompt, **kw)
		self.token_count = 0

	def __next__(self, *args, **kw):
		output_ids = super().__iter__(*args, **kw)
		self.token_count += output_ids.sequences.shape[1]
		return output_ids

class TransformersChatEngine:
	def __init__(self, model_name: str, device: str = None, fp16: bool = True, 
					output_json=True,
					gpus: int = 1):
		"""
		通用大模型加载器，支持 GPU 数量与编号控制
		:param model_name: 模型名称或路径
		:param device: 指定设备如 "cuda:0"，默认自动选择
		:param fp16: 是否使用 fp16 精度（适用于支持的 GPU）
		:param gpus: 使用的 GPU 数量，1 表示单卡，>1 表示多卡推理（使用 device_map='auto'）
		"""
		self.output_json = output_json
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.is_multi_gpu = gpus > 1 and torch.cuda.device_count() >= gpus

		print(f"✅ Using device: {self.device}, GPUs: {gpus}, Multi-GPU: {self.is_multi_gpu}")

		# Tokenizer 加载
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

		# 模型加载
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype=torch.float16 if fp16 and "cuda" in self.device else torch.float32,
			device_map="auto" if self.is_multi_gpu else None
		)

		if not self.is_multi_gpu:
			self.model.to(self.device)

		self.model.eval()
		self.is_chat = is_chat_model(model_name, self.tokenizer)
		if self.is_chat:
			self.messages = [ ]

		print(f'{self.model.generation_config=}')
	
	def set_system_prompt(self, prompt):
		if self.is_chat:
			self.messages = [{

				'role': 'system',
				'content': prompt
			}]
	def set_assistant_prompt(self, prompt):
		if self.is_chat:
			self.messages.append({
				'role': 'assistant',
				'content': prompt
			})
	def set_user_prompt(self, prompt):
		if self.is_chat:
			self.messages.append({
				'role': 'user',
				'content': prompt
			})
			return build_chat_prompt(self.messages)
		return prompt

	def generate(self, prompt: str):
		t1 = time()
		prompt = self.set_user_prompt(prompt)
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		output_ids = self.model.generate(
			**inputs,
			max_new_tokens=128,
			generation_config=self.model.generation_config
		)
		output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
		t2 = time
		text = output_text[len(prompt):] if output_text.startswith(prompt) else output_text
		self.set_assistant_prompt(text)
		if not self.output_json:
			return text
		input_tokens = inputs["input_ids"].shape[1]
		output_tokens = len(self.tokenizer(text, return_tensors="pt")["input_ids"][0])
		return {
			'content':text,
			'input_tokens': input_tokens,
			'output_tokens': output_tokens,
			'finish_time': t2 - t1,
			'response_time': t2 - t1
		}

	def stream_generate(self, prompt: str):
		t1 = time()
		prompt = self.set_user_prompt(prompt)
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
		input_tokens = inputs["input_ids"].shape[1]
		streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

		generation_kwargs = dict(
			**inputs,
			streamer=streamer,
			max_new_tokens=16000,
			generation_config=self.model.generation_config
		)

		thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
		thread.start()
		first = True
		all_txt = ''
		for new_text in streamer:
			all_txt += new_text
			if first:
				t2 = time()
				first = False
			if not self.output_json:
				yield new_text
			yield {
				'content': new_text,
				'done': False
			}
		output_tokens = len(self.tokenizer(all_txt, return_tensors="pt")["input_ids"][0])
		self.set_assistant_prompt(all_txt)
		t3 = time()
		if self.output_json:
			yield {
				'done': True,
				'content':'',
				'response_time': t2 - t1,
				'finish_time': t3 - t1,
				'input_tokens': input_tokens,
				'output_tokens': output_tokens 
			}

if __name__ == '__main__':
	import os
	import sys
	import argparse
	def parse_args():
		parser = argparse.ArgumentParser(description="Transformers Chat CLI")
		parser.add_argument("--model", type=str, required=True, help="模型路径或 Hugging Face 名称")
		parser.add_argument("--gpus", type=int, default=1, help="使用 GPU 数量")
		parser.add_argument("--stream", action="store_true", help="是否流式输出")
		return parser.parse_args()

	def print_content(outd):
		if isinstance(outd, dict):
			print(outd['content'], end="", flush=True)
		else:
			print(outd, end="", flush=True)

	def print_info(outd):
		if isinstance(outd, dict):
			if outd['done']:
				print(f"response_time={outd['response_time']}, finish_time={outd['finish_time']}, input_tokens={outd['input_tokens']}, output_tokens={outd['output_tokens']}\n")
		else:
			print('\n');

	def generate(engine, stream):
		while True:
			print('prompt("q" to exit):')
			p = input()
			if p == 'q':
				break
			if not p:
				continue
			if stream:
				for outd in engine.stream_generate(p):
					print_content(outd)
				print('\n')
				print_info(outd)
			else:
				outd = engine.generate(p)
				print_content(outd)
				print('\n')
				print__info(outd)

	def main():
		args = parse_args()
		print(f'{args=}')
		engine = TransformersChatEngine(
			model_name=args.model,
			gpus=args.gpus
		)
		generate(engine, args.stream)
	
	main()
