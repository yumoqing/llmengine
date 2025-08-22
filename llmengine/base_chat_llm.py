import threading
import asyncio
import json
import torch
from time import time
from aiostream import stream
from transformers import TextIteratorStreamer
from appPublic.log import debug
from appPublic.worker import awaitify
from appPublic.uniqueID import getID

model_pathMap = {
}
def llm_register(model_key, Klass):
	model_pathMap[model_key] = Klass

def get_llm_class(model_path):
	keys = [k for k in model_pathMap.keys()]
	keys.sort()
	keys.reverse()
	for k in keys:
		if len(model_path.split(k)) > 1:
			return model_pathMap[k]
	print(f'{model_pathMap=}')
	return None

class BaseChatLLM:
	def use_mps_if_prosible(self):
		if torch.backends.mps.is_available():
			device = torch.device("mps")
			self.model = self.model.to(device)

	def get_streamer(self):
		return TextIteratorStreamer(
			tokenizer=self.tokenizer,
			skip_special_tokens=True,
			skip_prompt=True
		)

	def output_generator(self, streamer):
		all_txt = ''
		t1 = time()
		i = 0
		id = f'chatllm-{getID()}'
		for txt in streamer:
			if txt == '':
				continue
			if i == 0:
				t2 = time()
			i += 1
			all_txt += txt
			yield {
				"id":id,
				"object":"chat.completion.chunk",
				"created": t1,
				"model":self.model_id,
				"choices":[
					{
						"index":0,
						"delta":{
							"role": "assistant",
							"content":txt
						},
						"finish_reason":None
					}
				]
			}
		t3 = time()
		t = all_txt
		unk = self.tokenizer(t, return_tensors="pt")
		output_tokens = len(unk["input_ids"][0])
		yield {
			"id":id,
			"object":"chat.completion.chunk",
			"created": t1,
			"model":self.model_id,
			"response_time": t2 - t1,
			"finish_time": t3 - t1,
			"output_tokens": output_tokens,
			"choices":[
				{
					"index":0,
					"delta":{
						"content":""
					},
					"finish_reason":"stop"
				}
			]
		}
		
	def build_kwargs(self, inputs, streamer):
		generate_kwargs = dict(
			**inputs,
			streamer=streamer,
			max_new_tokens=512,
			do_sample=True,
			eos_token_id=self.tokenizer.eos_token_id
		)
		return generate_kwargs

	def _messages2inputs(self, messages):
		debug(f'{messages=}')
		return self.processor.apply_chat_template(
			messages, add_generation_prompt=True, 
			tokenize=True,
			return_dict=True, return_tensors="pt"
		).to(self.model.device, dtype=torch.bfloat16)

	def _gen(self, messages):
		inputs = self._messages2inputs(messages)
		input_len = inputs["input_ids"].shape[-1]
		streamer = self.get_streamer()
		kwargs = self.build_kwargs(inputs, streamer)
		thread = threading.Thread(target=self.model.generate, 
						kwargs=kwargs)
		thread.start()
		for d in self.output_generator(streamer):
			if d['choices'][0]['finish_reason'] == 'stop':
				d['input_tokens'] = input_len
			yield d

	async def async_gen(self, messages):
		async for d in stream.iterate(self._gen(messages)):
			yield d
	
	async def chat_completion_stream(self, messages):
		async for d in self.async_gen(messages):
			if d['choices'][0]['finish_reason']:
				d['usage'] = {
					'prompt_tokens': d['input_tokens'],
					'completion_tokens': d['output_tokens'],
					'total_tokens': d['input_tokens'] + d['output_tokens']
				}
			s = f'data: {json.dumps(d)}\n'
			yield s
		yield 'data: [DONE]\n'
	
	def reference(self, messages):
		t1 = time()
		inputs = self._messages2inputs(messages)
		input_len = inputs["input_ids"].shape[-1]
		streamer = self.get_streamer()
		kwargs = self.build_kwargs(inputs, streamer)
		thread = threading.Thread(target=self.model.generate, 
						kwargs=kwargs)
		thread.start()
		txt = ''
		i_tokens = o_tokens = 0
		i = 0
		for d in self.output_generator(streamer):
			if i == 0:
				i = 1
				t1 = time()
			if d['choices'][0]['finish_reason'] != 'stop':
				txt += d['choices'][0]['delta']['content']
			else:
				o_tokens = d.get('output_tokens', 0)
		i_tokens  = input_len

		t2 = time()
		return {
			'id': f'chatcmpl-{getID()}',
			"object":"chat.completion",
			"created":t1,
			"model":self.model_id,
			"response_time": t2 - t1,
			"finish_time": t3 - t1,
			"choices":[
				{
					"index":0,
					"message":{
						"role": "assistant",
						"content": txt
					},
					"finish_reason":"stop"
				}
			],
			"usage": {
				"prompt_tokens": i_tokens,
				"completion_tokens": o_tokens,
				"total_tokens": i_tokens + o_tokens
			}
		}

	async def chat_completion(self, messages):
		f = awaitify(self.reference)
		return await f(messages)

