# m2m/fanyi_m2m.py
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from .base_m2m import BaseM2M, llm_register


class M2M100Translator(BaseM2M):
    def __init__(self, model_id, max_length=200):
        super().__init__(model_id)
        self.max_length = max_length
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_id)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,device_map="cuda"
        )
        self.model.eval()
        self.model_name = model_id.split('/')[-1]

    def m2m(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """翻译一段话"""
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                max_length=self.max_length
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# 注册模型（路径中包含 m2m100 即可匹配）
llm_register("m2m100", M2M100Translator)