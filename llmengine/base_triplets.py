from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

class BaseRelationLLM:

	def extract_triplets_typed(self, text):
		triplets = []
		relation = ''
		text = text.strip()
		current = 'x'
		subject, relation, object_, object_type, subject_type = '','','','',''

		for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
			if token == "<triplet>" or token == "<relation>":
				current = 't'
				if relation != '':
					triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
					relation = ''
				subject = ''
			elif token.startswith("<") and token.endswith(">"):
				if current == 't' or current == 'o':
					current = 's'
					if relation != '':
						triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
					object_ = ''
					subject_type = token[1:-1]
				else:
					current = 'o'
					object_type = token[1:-1]
					relation = ''
			else:
				if current == 't':
					subject += ' ' + token
				elif current == 's':
					object_ += ' ' + token
				elif current == 'o':
					relation += ' ' + token
		if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
			triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
		return triplets

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

	def extract_triplets(self, text):
		inputs = build_inputs(text)
		preds = gen_preds(inputs)
		
		# Extract triplets
		triplets = []
		for idx, sentence in enumerate(decoded_preds):
			x = self.extract_triplets_typed(sentence)
			triplets += x
		print(triplets)

