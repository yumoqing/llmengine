import os
import torch
import re
import traceback
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from appPublic.log import debug, error, warning, info
from appPublic.worker import awaitify
from base_triple import BaseTripleExtractor, llm_register

class MRebelTripleExtractor(BaseTripleExtractor):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        try:
            debug(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            debug(f"Loading model from {model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.device = self.use_mps_if_possible()
            if self.device.type == "cuda":
                self.model = self.model.to(dtype=torch.float16)
                debug("Model converted to FP16 for CUDA")
            self.triplet_id = self.tokenizer.convert_tokens_to_ids("<triplet>")
            debug(f"Loaded mREBEL model, triplet_id: {self.triplet_id}")
            if self.device.type == "cuda":
                debug(f"GPU memory allocated after model load: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
        except Exception as e:
            error(f"Failed to load mREBEL model: {str(e)}")
            raise RuntimeError(f"Failed to load mREBEL model: {str(e)}")

        self.gen_kwargs = {
            "max_length": 256,
            "min_length": 10,
            "length_penalty": 0.5,
            "num_beams": 3,
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "decoder_start_token_id": self.triplet_id,
        }

    def extract_triplets_typed(self, text: str) -> list:
        """Parse mREBEL generated text for triplets."""
        triplets = []
        debug(f"Raw generated text: {text}")

        tokens = []
        in_tag = False
        buffer = ""
        for char in text:
            if char == '<':
                in_tag = True
                if buffer:
                    tokens.append(buffer.strip())
                    buffer = ""
                buffer += char
            elif char == '>':
                in_tag = False
                buffer += char
                tokens.append(buffer.strip())
                buffer = ""
            else:
                buffer += char
        if buffer:
            tokens.append(buffer.strip())

        special_tokens = ["<s>", "<pad>", "</s>", "tp_XX", "__en__", "__zh__", "zh_CN"]
        tokens = [t for t in tokens if t not in special_tokens and t]
        debug(f"Processed tokens: {tokens}")

        i = 0
        while i < len(tokens):
            if tokens[i] == "<triplet>" and i + 5 < len(tokens):
                entity1 = tokens[i + 1]
                type1 = tokens[i + 2][1:-1] if tokens[i + 2].startswith("<") and tokens[i + 2].endswith(">") else ""
                entity2 = tokens[i + 3]
                type2 = tokens[i + 4][1:-1] if tokens[i + 4].startswith("<") and tokens[i + 4].endswith(">") else ""
                relation = tokens[i + 5]

                if entity1 and type1 and entity2 and type2 and relation:
                    triplets.append({
                        'head': entity1.strip(),
                        'head_type': type1,
                        'type': relation.strip(),
                        'tail': entity2.strip(),
                        'tail_type': type2
                    })
                    debug(f"Added triplet: {entity1}({type1}) - {relation} - {entity2}({type2})")
                i += 6
            else:
                i += 1

        return triplets

    async def extract_triplets(self, text: str) -> list:
        """Extract triplets from text, splitting into sub-chunks by .; and \n."""
        try:
            if not text:
                raise ValueError("Text cannot be empty")

            # 按 .、；、\n 分割文本为子片段
            sub_texts = re.split(r'[.;\n]+', text)
            sub_texts = [sub.strip() for sub in sub_texts if sub.strip() and len(sub.strip()) >= 10]
            debug(f"Split text into {len(sub_texts)} sub-chunks: {[sub[:50] for sub in sub_texts[:5]]}")
            token_lengths = [len(self.tokenizer(sub, add_special_tokens=False)['input_ids']) for sub in sub_texts]
            debug(f"Sub-chunk token lengths: {token_lengths}")
            if any(length > 256 for length in token_lengths):
                warning(f"Some sub-chunks exceed max_length=256: {token_lengths}")

            # 记录开始时的 GPU 内存
            if self.device.type == "cuda":
                debug(f"GPU memory allocated before processing chunk: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")

            # 批量处理子片段
            batch_size = 10
            triplets = []
            for batch_start in range(0, len(sub_texts), batch_size):
                batch_texts = sub_texts[batch_start:batch_start + batch_size]
                debug(f"Processing batch {batch_start // batch_size + 1} with {len(batch_texts)} sub-chunks")

                # 批量编码
                model_inputs = self.tokenizer(
                    batch_texts,
                    max_length=256,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                try:
                    with torch.cuda.amp.autocast():
                        generated_tokens = self.model.generate(
                            model_inputs["input_ids"],
                            attention_mask=model_inputs["attention_mask"],
                            **self.gen_kwargs
                        )
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    for idx, sentence in enumerate(decoded_preds):
                        debug(f"Sub-chunk {batch_start + idx + 1} generated text: {sentence[:50]}...")
                        chunk_triplets = self.extract_triplets_typed(sentence)
                        if chunk_triplets:
                            debug(f"Sub-chunk {batch_start + idx + 1} extracted {len(chunk_triplets)} triplets: {chunk_triplets}")
                            triplets.extend(chunk_triplets)
                except Exception as e:
                    warning(f"Error processing batch {batch_start // batch_size + 1}: {str(e)}")
                    continue

            # 记录结束时的 GPU 内存
            if self.device.type == "cuda":
                debug(f"GPU memory allocated after processing chunk: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
                torch.cuda.empty_cache()
                debug(f"GPU memory cleared after processing chunk")

            debug(f"Total extracted {len(triplets)} triplets from {len(sub_texts)} sub-chunks")
            return triplets

        except Exception as e:
            error(f"Failed to extract triplets: {str(e)}")
            debug(f"Traceback: {traceback.format_exc()}")
            return []

llm_register("mrebel-large", MRebelTripleExtractor)