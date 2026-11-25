"""OpenVINO 推理后端：同一份模型在 Intel CPU / iGPU 上跑（可插拔）。

为什么需要它（2026-11-25 在 nuc115 / i5-1135G7 实测）：

- 旧的 bgeembedding.encode() 是「逐条 embed_query」，N 条文本 = N 次 forward，
  每次都要把 ~1.2 GB 有效权重从内存重读一遍，正好压在 DDR 带宽上。
  1 条 134 ms / 32 条 4202 ms —— 完全线性，等于没有批处理。
- 同一台机的 Iris Xe iGPU 一直空着。转成 OpenVINO IR（fp16，1.1 GB）后用 GPU 插件：
  1 条 39 ms / 32 条 314 ms / 903 token 长文本 589 ms，进程 RSS 从 10.2 GB 掉到 ~1.2 GB
  （旧进程 10.2 GB 曾把 2 GB swap 吃满，拖慢同机 doit / MariaDB）。
- 数值：与 torch fp32 输出余弦 min 0.99971 / mean 0.999964，top3 近邻 11/11 一致。
- Tiger Lake 没有 NPU（只有 GNA，OpenVINO 已移除 GNA 插件），所以只做 CPU / iGPU 双后端。

几个必须遵守的坑：
1. 混长 batch 直接 padding 会把算力白烧掉（11 条混长：一次性 4.37 s，按长度分桶 1.38 s）。
   所以这里按长度排序后切桶，并同时受 max_batch 与 token 预算两个上限约束。
2. 不同 seq_length 不需要分桶对齐形状，OpenVINO GPU 动态 shape 首次调用没有额外编译惩罚
   （seq 18→1698 首/次调用基本一致），不要为此去固定 shape。
3. CPU 用 f32（Tiger Lake 无 AVX512-FP16），GPU 用 f16。
4. 输出已在图内做过 CLS pooling + L2 归一化（与 sentence-transformers 的
   normalize_embeddings=True 一致，norm=1.0000），后处理不要再归一化一次。
"""

import json
import os
import threading

import numpy as np

DEFAULT_MAX_BATCH = 32
DEFAULT_TOKEN_BUDGET = 8192


def _resolve_max_length(tokenizer, config, explicit=None):
	"""取模型真实上限：显式入参 / tokenizer / config 三者取最小。

	tokenizer.model_max_length 大于 1e6 表示未设置（transformers 的哨兵值），必须忽略。
	硬编码上限导致过 500 的事故见 llmengine-service-operations 技能 §3。
	"""
	limits = []
	if explicit:
		limits.append(int(explicit))
	tk_max = getattr(tokenizer, 'model_max_length', None)
	if tk_max and tk_max < 10 ** 6:
		limits.append(int(tk_max))
	cfg_max = getattr(config, 'max_position_embeddings', None)
	if cfg_max and 0 < cfg_max < 10 ** 6:
		limits.append(int(cfg_max))
	if not limits:
		return 512
	return min(limits)


def _pooling_is_cls(model_id):
	"""bge-m3 用 CLS pooling；从 sentence-transformers 的 1_Pooling/config.json 读，读不到按 CLS。"""
	try:
		cfg = json.load(open(os.path.join(model_id, '1_Pooling', 'config.json')))
	except Exception:
		return True
	if cfg.get('pooling_mode_mean_tokens'):
		return False
	return True


def _torch_export(model_id, max_length):
	"""没有现成 ONNX 时的退路：从 torch 权重导出成 OpenVINO 图（输出契约与 ONNX 版一致）。"""
	import torch
	from transformers import AutoModel, AutoTokenizer

	tk = AutoTokenizer.from_pretrained(model_id)
	model = AutoModel.from_pretrained(model_id)
	model.eval()
	use_cls = _pooling_is_cls(model_id)

	class Wrapper(torch.nn.Module):
		def __init__(self, m, cls_pool):
			super().__init__()
			self.m = m
			self.cls_pool = cls_pool

		def forward(self, input_ids, attention_mask):
			out = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
			if self.cls_pool:
				pooled = out[:, 0]
			else:
				mask = attention_mask.unsqueeze(-1).float()
				pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
			pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
			return out, pooled

	example = tk(['warmup'], padding=True, truncation=True, max_length=min(max_length, 128),
				 return_tensors='pt')
	import openvino as ov
	with torch.no_grad():
		ov_model = ov.convert_model(Wrapper(model, use_cls), example_input=dict(example))
	return ov_model


class OvEncoder:
	"""OpenVINO 版本的文本向量化器：批处理 + 长度分桶 + 真实 token 计数。"""

	def __init__(self, model_id, device=None, max_length=None):
		import openvino as ov
		from transformers import AutoConfig, AutoTokenizer

		self.model_id = model_id
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
		self.config = AutoConfig.from_pretrained(model_id)
		self.max_length = _resolve_max_length(self.tokenizer, self.config, max_length)

		self.core = ov.Core()
		self.device = self._resolve_device(self.core, device)
		ir_xml = self._ensure_ir(ov)

		props = {'PERFORMANCE_HINT': 'THROUGHPUT'}
		props['INFERENCE_PRECISION_HINT'] = 'f32' if self.device == 'CPU' else 'f16'
		self.compiled = self.core.compile_model(ir_xml, self.device, props)

		outs = list(self.compiled.outputs)
		self.pooled_port = out_port = None
		for i, o in enumerate(outs):
			names = set(o.get_names()) | {o.get_any_name()}
			if 'sentence_embedding' in names:
				self.pooled_port = i
		self.pooled_port = 1 if self.pooled_port is None else self.pooled_port
		self._local = threading.local()

	@property
	def device_name(self):
		return self.device

	def _resolve_device(self, core, device):
		want = (device or os.environ.get('LLMENGINE_EMBED_DEVICE') or 'AUTO').upper()
		avail = core.available_devices
		if want == 'AUTO':
			want = 'GPU' if 'GPU' in avail else 'CPU'
		if want.startswith('GPU'):
			if not any(d.startswith('GPU') for d in avail):
				want = 'CPU'
		return want

	def _ensure_ir(self, ov):
		"""IR 缓存到 <model_dir>/ov_ir/，优先用已有 ONNX 转（10 s，不必再跑 torch 导出）。"""
		ir_dir = os.path.join(self.model_id, 'ov_ir')
		xml = os.path.join(ir_dir, 'model.xml')
		if os.path.exists(xml) and os.path.getsize(xml) > 0:
			return xml
		os.makedirs(ir_dir, exist_ok=True)
		onnx_path = os.path.join(self.model_id, 'onnx', 'model.onnx')
		if os.path.exists(onnx_path):
			model = ov.convert_model(onnx_path)
		else:
			model = _torch_export(self.model_id, self.max_length)
		tmp = xml + '.tmp.xml'
		ov.save_model(model, tmp, compress_to_fp16=True)
		os.replace(tmp, xml)
		return xml

	def _request(self):
		req = getattr(self._local, 'req', None)
		if req is None:
			req = self.compiled.create_infer_request()
			self._local.req = req
		return req

	def encode(self, texts, max_batch=None, token_budget=None):
		"""返回 (vectors: list[list[float]], prompt_tokens: int)。"""
		texts = [t if isinstance(t, str) else str(t) for t in texts]
		if not texts:
			return [], 0
		max_batch = int(max_batch or os.environ.get('LLMENGINE_EMBED_MAX_BATCH') or DEFAULT_MAX_BATCH)
		token_budget = int(token_budget or os.environ.get('LLMENGINE_EMBED_TOKEN_BUDGET')
						   or DEFAULT_TOKEN_BUDGET)

		enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=False)
		ids_list = enc['input_ids']
		lens = [max(len(x), 1) for x in ids_list]
		order = sorted(range(len(texts)), key=lambda i: lens[i])

		vectors = [None] * len(texts)
		req = self._request()
		pos = 0
		while pos < len(order):
			bucket = []
			maxlen = 0
			while pos < len(order):
				j = order[pos]
				newmax = max(maxlen, lens[j])
				# 桶内最长的长度 × 条数 = padding 后的实际 token 数，用它守预算
				if bucket and (len(bucket) >= max_batch
							   or newmax * (len(bucket) + 1) > token_budget):
					break
				bucket.append(j)
				maxlen = newmax
				pos += 1
			feats = self.tokenizer.pad(
				[{'input_ids': ids_list[j], 'attention_mask': [1] * lens[j]} for j in bucket],
				padding=True, return_tensors='np')
			out = req.infer({0: feats['input_ids'].astype(np.int64),
							 1: feats['attention_mask'].astype(np.int64)})
			vecs = out[self.pooled_port]
			for k, j in enumerate(bucket):
				vectors[j] = np.asarray(vecs[k], dtype=np.float32).tolist()

		return vectors, int(sum(lens))