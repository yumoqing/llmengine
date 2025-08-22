#!/share/vllm-0.8.5/bin/python

# pip install accelerate
from appPublic.worker import awaitify
from appPublic.log import debug
from ahserver.serverenv import get_serverenv
from PIL import Image
import torch
from llmengine.base_chat_llm import BaseChatLLM, llm_register
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaichuanM2LLM(BaseChatLLM):
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
            max_new_tokens=4096
        )
        return generate_kwargs

	def _messages2inputs(self, messages):
		text = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			thinking_mode='on' # on/off/auto 
		)
		model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
		return model_inputs

llm_register('Baichuan-M2', BaichuanM2LLM)
