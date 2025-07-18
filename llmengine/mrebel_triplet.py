
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from base_triplets import BaseTriplets, llm_register

class MrebelTriplets(BaseTriplets):
	def __init__(self, model_id):
		if 'mrebel' not in model_id:
			raise Exception(f'{model_id} is not a mrebel model')

		# Load model and tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, 
							src_lang="zh_XX", tgt_lang="tp_XX")
		# Here we set English ("en_XX") as source language. 
		# To change the source language swap the first token of the 
		# input for your desired language or change to supported language. 
		# For catalan ("ca_XX") or greek ("el_EL") 
		# (not included in mBART pretraining) you need a workaround:
		# tokenizer._src_lang = "ca_XX"
		# tokenizer.cur_lang_code_id = tokenizer.convert_tokens_to_ids("ca_XX")
		# tokenizer.set_src_lang_special_tokens("ca_XX")
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
		self.model_id = model_id
		self.model_name = model_id.split('/')[-1]
		self.gen_kwargs = {
			"max_length": 256,
			"length_penalty": 0,
			"num_beams": 3,
			"num_return_sequences": 3,
			"forced_bos_token_id": None,
		}

	def build_inputs(self, text):
		# Tokenizer text
		return self.tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
	
	def gen_preds(self, inputs):
		# Generate
		generated_tokens = self.model.generate(
			inputs['input_ids'].to(self.model.device)
			attention_mask=inputs["attention_mask"].to(self.model.device),
			decoder_start_token_id = self.tokenizer.convert_tokens_to_ids("tp_XX"),
			**self.gen_kwargs
		)
		# Extract text
		decoded_preds = self.tokenizer.batch_decode(generated_tokens, 
							skip_special_tokens=False)
		return decoded_preds

llm_register('mrebel', MrebelTriplets)
