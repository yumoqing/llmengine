# embed_all_unified.py
"""
Unified multimodal embedder (text, image, video, audio)
Features:
- All modalities mapped to the same embedding space (CLIP or CLAP)
- GPU/CPU/MPS auto detection
- FP16 autocast for speed
- Batch processing
- Video frame sampling + average pooling
- Audio resampling + CLAP embedding
- L2 normalized output for similarity search
"""

import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import av
import librosa
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from appPublic.jsonConfig import getConfig
from appPublic.worker import awaitify
from ahserver.webapp import webapp
from ahserver.serverenv import ServerEnv

try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception:
    FACE_LIB_AVAILABLE = False

# ------------------- Configuration -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

# Unified model for all modalities
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
FRAME_SAMPLE_RATE = 1.0  # fps for video
FRAME_LIMIT = 64
AUDIO_SR = 16000  # resample audio

# ------------------- Load model -------------------
from transformers import CLIPProcessor, CLIPModel

# ------------------- Utils -------------------

def deduplicate_faces(face_embeddings, eps=0.4, min_samples=2):
    emb_norm = normalize(face_embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(emb_norm)
    unique_faces = []
    for label in set(clustering.labels_):
        if label == -1:  # 噪声
            continue
        cluster_embs = emb_norm[clustering.labels_ == label]
        unique_faces.append(np.mean(cluster_embs, axis=0))
    return np.array(unique_faces)

def l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class MM_Embedding:
	def __init__(self, model_name):
		self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
		self.processor = CLIPProcessor.from_pretrained(model_name)
		if USE_FP16:
			self.model.half()

	# ------------------- Image -------------------
	def embed_images(self, paths, batch_size=16):
		results = {}
		for batch in chunked(paths, batch_size):
			imgs = [Image.open(p).convert("RGB") for p in batch]
			inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
			with torch.no_grad():
				if USE_FP16:
					with torch.cuda.amp.autocast():
						feats = self.model.get_image_features(**inputs)
				else:
					feats = self.model.get_image_features(**inputs)
			feats = feats.cpu().numpy()
			faces_list = []
			for img in imgs:
				faces = self.extract_faces(img)
				face_vecs = self.embed_faces(img)
				faces_list.append([faces, face_vecs])

			for p, v, fs in zip(batch, feats, faces_list):
				results[p] = {
					'type':'image',
					'path': p,
					'faces': fs[0],
					'face_vecs': fs[1],
					'face_count':len(fs[0]),
					'vector': l2_normalize(v)
				}
		return results

	# ------------------- Text -------------------
	def embed_texts(self, texts, batch_size=64):
		results = {}
		for batch in chunked(texts, batch_size):
			inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
			with torch.no_grad():
				if USE_FP16:
					with torch.cuda.amp.autocast():
						feats = self.model.get_text_features(**inputs)
				else:
					feats = self.model.get_text_features(**inputs)
			feats = feats.cpu().numpy()
			for t, v in zip(batch, feats):
				results[t] = l2_normalize(v)
		return results

	# ------------------- Video -------------------
	def embed_videos(self, paths, frame_rate=FRAME_SAMPLE_RATE, frame_limit=FRAME_LIMIT):
		results = {}
		for p in paths:
			container = av.open(p)
			frames = []
			fps = float(container.streams.video[0].average_rate) if container.streams.video else 30.0
			step = max(1, int(fps / max(1, frame_rate)))
			count = 0
			for i, frame in enumerate(container.decode(video=0)):
				if i % step == 0:
					frames.append(frame.to_image().convert("RGB"))
					count += 1
					if count >= frame_limit:
						break
			container.close()
			if not frames:
				results[p] = None
				continue
			# batch embed
			emb_list = []
			for batch in chunked(frames, 16):
				inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
				with torch.no_grad():
					if USE_FP16:
						with torch.cuda.amp.autocast():
							feats = self.model.get_image_features(**inputs)
					else:
						feats = self.model.get_image_features(**inputs)
				emb_list.append(feats.cpu().numpy())
			emb_array = np.vstack(emb_list)
			video_vec = l2_normalize(emb_array.mean(axis=0))
			face_vecs = 
			results[p] = video_vec
		return results

	# ------------------- Audio -------------------
	def embed_audios(self, paths, batch_size=4):
		results = {}
		for p in paths:
			y, sr = librosa.load(p, sr=AUDIO_SR, mono=True)
			# convert to mel spectrogram image
			S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
			S_db = librosa.power_to_db(S, ref=np.max)
			img = Image.fromarray(np.uint8((S_db - S_db.min())/(S_db.max()-S_db.min()+1e-9)*255)).convert("RGB").resize((224,224))
			inputs = self.processor(images=img, return_tensors="pt").to(DEVICE)
			with torch.no_grad():
				if USE_FP16:
					with torch.cuda.amp.autocast():
						feat = self.model.get_image_features(**inputs)
				else:
					feat = self.model.get_image_features(**inputs)
			results[p] = l2_normalize(feat.cpu().numpy()[0])
		return results

    def extract_faces(self, img: Image.Image):
        """返回裁剪后的人脸区域列表"""
        arr = np.array(img)
        face_locs = face_recognition.face_locations(arr)
        faces = []
        for (top, right, bottom, left) in face_locs:
            face = arr[top:bottom, left:right]
            faces.append(Image.fromarray(face))
        return faces

    def embed_faces(self, img: Image.Image):
        """提取人脸向量（face_recognition + CLIP）"""
        arr = np.array(img)
        encodings = face_recognition.face_encodings(arr)
        if not encodings:
            return []
        return [l2_normalize(np.array(e)) for e in encodings]

	# ------------------- Dispatcher -------------------
	def embed_batch(self, inputs):
		groups = {"image":[], "video":[], "audio":[], "text":[]}
		for item in inputs:
			p = Path(item)
			ext = item.lower()
			if p.exists():
				if any(ext.endswith(e) for e in [".jpg",".jpeg",".png",".bmp",".webp",".heic"]):
					groups["image"].append(item)
				elif any(ext.endswith(e) for e in [".mp4",".mov",".avi",".mkv"]):
					groups["video"].append(item)
				elif any(ext.endswith(e) for e in [".mp3",".wav",".flac"]):
					groups["audio"].append(item)
				else:
					groups["text"].append(item)
			else:
				groups["text"].append(item)
		outputs = {}
		if groups["image"]:
			outputs.update(embed_images(groups["image"]))
		if groups["video"]:
			outputs.update(embed_videos(groups["video"]))
		if groups["audio"]:
			outputs.update(embed_audios(groups["audio"]))
		if groups["text"]:
			outputs.update(embed_texts(groups["text"]))
		return outputs

def init():
	env = ServerEnv()
	config = getConfig()
	env.mm_model = MM_Embedding(config.model_name)
	env.embeded_batch = awaitify(env.mm_model.embeded_batch)
# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="file paths or text strings")
    parser.add_argument("--out", default="embeddings.npy")
    args = parser.parse_args()

    embeddings = embed_batch(args.inputs)
	# save dict of name->vector
    out_dict = {k:v.tolist() for k,v in embeddings.items()}
    np.save(args.out, out_dict)
    print(f"Saved embeddings to {args.out}")

