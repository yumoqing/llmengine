import threading
import asyncio
import json
import torch
from time import time
from transformers import TextIteratorStreamer
from appPublic.log import debug
from appPublic.worker import awaitify
from appPublic.uniqueID import getID

model_pathMap = {
}
def llm_register(model_key, Klass):
	model_pathMap[model_key] = Klass

def get_llm_class(model_path):
	for k,klass in model_pathMap.items():
		if len(model_path.split(k)) > 1:
			return klass
	print(f'{model_pathMap=}')
	return None

class BaseChatLLM:
	def use_mps_if_prosible(self):
		if torch.backends.mps.is_available():
			device = torch.device("mps")
			self.model = self.model.to(device)

	def get_session_key(self):
		return self.model_id + ':messages'

	def _get_session_messages(self, session):
		key = self.get_session_key()
		messages = session.get(key) or []
		return messages

	def _set_session_messages(self, session, messages):
		key = self.get_session_key()
		session[key] = messages

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
		id = f'chatllm-{getID}'
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
				"created":time(),
				"model":self.model_id,
				"choices":[
					{
						"index":0,
						"delta":{
							"content":txt
						},
						"logprobs":None,
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
			"created":time(),
			"model":self.model_id,
			"response_time": t2 - t1,
			"finish_time": t3 - t1,
			"output_token": output_tokens,
			"choices":[
				{
					"index":0,
					"delta":{
						"content":""
					},
					"logprobs":None,
					"finish_reason":"stop"
				}
			]
		}
		
	def _generator(self, session, prompt, image_path, video_path, audio_path, sys_prompt):
		messages = self._get_session_messages(session)
		if sys_prompt:
			messages.append(self._build_sys_message(sys_prompt))
		messages.append(self._build_user_message(prompt, image_path=image_path))
		# debug(f'{messages=}')
		all_txt = ''
		for d in self._gen(messages):
			if d['choices'][0]['finish_reason'] == 'stop':
				messages.append(self._build_assistant_message(all_txt))
			else:
				all_txt += d['choices'][0]['delta']['content']
			yield d
		self._set_session_messages(session, messages)

	async def _async_generator(self, session, prompt, image_path, video_path, audio_path, sys_prompt):
		for d in self._generator(session, prompt, image_path, video_path, audio_path, sys_prompt):
			await asyncio.sleep(0)
			yield d

	def generate(self, session, prompt,
						image_path=None, 
						video_path=None, 
						audio_path=None, 
						sys_prompt=None):
		for d in self._generator(session, prompt, image_path, video_path, audio_path, sys_prompt):
			if d['choices'][0]['finish_reason'] == 'stop':
				return d
	def stream_generate(self, session, prompt,
						image_path=None, 
						video_path=None, 
						audio_path=None, 
						sys_prompt=None):
		for d in self._generator(session, prompt, image_path, video_path, audio_path, sys_prompt):
			s = f'data: {json.dumps(d)}\n'
			yield s

	async def async_generate(self, session, prompt,
                        image_path=None, 
                        video_path=None,
                        audio_path=None,
                        sys_prompt=None):
		async for d in self._async_generator(session, prompt, image_path, video_path, audio_path, sys_prompt):
			await asyncio.sleep(0)
			if d['choices'][0]['finish_reason'] == 'stop':
				return d

	async def async_stream_generate(self, session, prompt,
                        image_path=None,
                        video_path=None,
                        audio_path=None,
                        sys_prompt=None):
		async for d in self._async_generator(session, prompt, image_path, video_path, audio_path, sys_prompt):
			s = f'data: {json.dumps(d)}\n'
			yield s
		yield 'data: [DONE]'

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

class T2TChatLLM(BaseChatLLM):
	def _build_assistant_message(self, prompt):
		return {
			"role":"assistant",
			"content":prompt
		}

	def _build_sys_message(self, prompt):
		return {
			"role":"system",
			"content": prompt
		}

	def _build_user_message(self, prompt, **kw):
		return {
			"role":"user",
			"content": prompt
		}

class MMChatLLM(BaseChatLLM):
	""" multiple modal chat LLM """
	def _build_assistant_message(self, prompt):
		return {
			"role":"assistant",
			"content":[{"type": "text", "text": prompt}]
		}

	def _build_sys_message(self, prompt):
		return {
			"role":"system",
			"content":[{"type": "text", "text": prompt}]
		}

	def _build_user_message(self, prompt, image_path=None, 
					video_path=None, audio_path=None):
		contents = [
			{
				"type":"text", "text": prompt
			}
		]
		if image_path:
			contents.append({
				"type": "image",
				"image": image_path
			})
		if video_path:
			contents.append({
				"type": "video",
				"video":video_path
			})
		if audio_path:
			contents.append({
				"tyoe": "audio",
				"audio": audio_path
			})
		return {
				"role": "user",
				"content": contents
		}
	
