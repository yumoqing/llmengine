# for model mistralai/Devstral-Small-2505
from appPublic.worker import awaitify
from appPublic.log import debug
from ahserver.serverenv import get_serverenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from mistral_common.protocol.instruct.messages import (
	SystemMessage, UserMessage, AssistantMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

import torch
from llmengine.base_chat_llm import BaseChatLLM, T2TChatLLM, llm_register

class DevstralLLM(T2TChatLLM):
	def __init__(self, model_id):
		tekken_file = f'{model_id}/tekken.json'
		self.tokenizer = MistralTokenizer.from_file(tekken_file)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			torch_dtype="auto",
			device_map="auto"
		)
		self.model_id = model_id
		
	def _build_assistant_message(self, prompt):
		return AssistantMessage(content=prompt)

	def _build_sys_message(self, prompt):
		return SystemMessage(content=prompt)

	def _build_user_message(self, prompt, **kw):
		return UserMessage(content=prompt)

	def get_streamer(self):
		return TextIteratorStreamer(
			tokenizer=self.tokenizer,
			skip_prompt=True
		)

	def build_kwargs(self, inputs, streamer):
		generate_kwargs = dict(
			**inputs, 
			streamer=streamer,
			max_new_tokens=32768,
			do_sample=True
		)
		return generate_kwargs

	def _messages2inputs(self, messages):
		tokenized = self.tokenizer.encode_chat_completion(
			ChatCompletionRequest(messages=messages)
		)
		return {
			'input_ids': torch.tensor([tokenized.tokens])
		}

llm_register('mistralai/Devstral', DevstralLLM)

