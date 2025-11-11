import os
import argparse
import numpy as np
import torch
from PIL import Image
import av
import librosa
from pathlib import Path
from math import ceil
from traceback import format_exc

from appPublic.jsonConfig import getConfig
from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, exception

from ahserver.serverenv import ServerEnv
from ahserver.webapp import webserver

from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import base64
import traceback
import uuid

helptext = """CLIP 多模态统一嵌入服务
API 地址：
POST http://localhost:8883/v1/embed
功能：
将 文本 / 图片 / 视频 / 音频 统一转为 1024 维 L2 归一化向量
支持人脸检测 + 去重（基于 DBSCAN 余弦距离）

输入格式 (JSON):
{
  "inputs": [
    "文本字符串",
    "/path/to/image.jpg",
    "/path/to/video.mp4",
    "/path/to/audio.wav"
  ]
}

输出格式:
{
  "data": { ... },
  "object": "embedding.result",
  "model": "CLIP-ViT-H-14-laion2B-s32B-b79K"
}

特性：
- 自动识别文件类型（文本/图像/音频/视频）
- 视频抽帧 + 人脸去重
- 文件不存在 → 自动降级为文本嵌入
- 所有向量 L2 归一化，可直接余弦相似度
- 人脸向量支持聚类去重（eps=0.4）

文档查看：
GET http://localhost:8883/docs
"""
try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception:
    debug('人脸识别库导入失败')
    FACE_LIB_AVAILABLE = False

# ------------------- 配置 -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
CLIP_MODEL_NAME = "/data/ymq/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
FRAME_SAMPLE_RATE = 1.0
FRAME_LIMIT = 64
AUDIO_SR = 16000
IMAGE_DIR = Path("/share/wangmeihua/data/mmembedding/image")
AUDIO_DIR = Path("/share/wangmeihua/data/mmembedding/audio")
VIDEO_DIR = Path("/share/wangmeihua/data/mmembedding/video")

for d in [IMAGE_DIR, AUDIO_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------- 工具函数 -------------------
def l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def deduplicate_faces(face_embeddings, eps=0.4, min_samples=2):
    if len(face_embeddings) == 0:
        return np.array([])
    emb_norm = normalize(face_embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(emb_norm)
    unique = []
    for label in set(clustering.labels_):
        if label == -1:
            continue
        cluster = emb_norm[clustering.labels_ == label]
        unique.append(np.mean(cluster, axis=0))
    return np.array(unique)

# ------------------- 主模型类 -------------------
class MM_Embedding:
    def __init__(self, model_name):
        debug(f"Loading CLIP model: {model_name}")
        self.model_name = Path(model_name).name
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            device_map="auto" if USE_FP16 else None
        ).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_batch(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a list")
        if len(inputs) == 0:
            return {}

        groups = {"image": [], "video": [], "audio": [], "text": []}
        results = {}
        for item in inputs:
            # ------------------- 新格式：字典 -------------------
            if isinstance(item, dict):
                typ = item.get("type")
                data_uri = item.get("data")

                if typ == "text":
                    content = item.get("content", "")
                    if content:
                        groups["text"].append(content)
                    continue

                if typ in {"image", "video", "audio"} and data_uri:
                    try:
                        # 1. 提取 base64
                        try:
                            header, b64 = data_uri.split(",", 1)
                            debug(f"header: {header},b64: {b64}")
                            binary = base64.b64decode(b64)
                        except Exception as e:
                            error(f"解码失败: {str(e)}, 堆栈: {traceback.format_exc()}")

                        # 2. 确定扩展名
                        mime_to_ext = {
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "image/webp": ".webp",
                            "video/mp4": ".mp4",
                            "video/webm": ".webm",
                            "audio/mpeg": ".mp3",
                            "audio/wav": ".wav",
                            "audio/ogg": ".ogg",
                        }
                        mime = header.split(";")[0].split(":")[1]
                        ext = mime_to_ext.get(mime, ".bin")

                        # 3. 生成唯一文件名 + 存储
                        uid = uuid.uuid4().hex[:12]
                        if typ == "image":
                            save_dir = IMAGE_DIR
                            fake_path = save_dir / f"{uid}{ext}"
                            fake_path = str(fake_path)
                        elif typ == "video":
                            save_dir = VIDEO_DIR
                            fake_path = save_dir / f"{uid}{ext}"
                            fake_path = str(fake_path)
                        elif typ == "audio":
                            save_dir = AUDIO_DIR
                            fake_path = save_dir / f"{uid}{ext}"
                            fake_path = str(fake_path)

                        Path(fake_path).write_bytes(binary)
                        debug(f"保存多媒体文件: {fake_path} ({len(binary) / 1024 / 1024:.2f}MB)")

                        # 4. 放入对应 group（CLIP 直接用路径）
                        if typ == "image":
                            groups["image"].append(fake_path)
                        elif typ == "video":
                            groups["video"].append(fake_path)
                        elif typ == "audio":
                            groups["audio"].append(fake_path)

                        # 记录原始来源（可选）
                        results[fake_path] = {"type": typ, "source": "data_uri", "original_mime": mime}
                        continue

                    except Exception as e:
                        results[id(item)] = {"type": "error", "error": f"data URI 解码失败: {e}"}
                        continue

        if groups["image"]:
            results.update(self._embed_images(groups["image"]))
        if groups["video"]:
            results.update(self._embed_videos(groups["video"]))
        if groups["audio"]:
            results.update(self._embed_audios(groups["audio"]))
        if groups["text"]:
            results.update(self._embed_texts(groups["text"]))
        debug(f"最终返回结果是：{results}")
        return results

    def _embed_texts(self, texts):
        results = {}
        for batch in chunked(texts, 64):
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=USE_FP16):
                    feats = self.model.get_text_features(**inputs)
            feats = feats.cpu().numpy()
            for t, v in zip(batch, feats):
                results[t] = {"type": "text", "vector": l2_normalize(v).tolist()}
        return results

    def _embed_images(self, paths):
        results = {}
        for batch in chunked(paths, 16):
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=USE_FP16):
                    feats = self.model.get_image_features(**inputs)
            feats = feats.cpu().numpy()
            for p, v, img in zip(batch, feats, imgs):
                face_vecs, _ = self._detect_faces(img)
                results[p] = {
                    "type": "image",
                    "path": p,
                    "vector": l2_normalize(v).tolist(),
                    "face_count": len(face_vecs),
                    "face_vecs": [vec.tolist() for vec in face_vecs]
                }
        return results

    def _embed_videos(self, paths):
        results = {}
        for p in paths:
            try:
                container = av.open(p)
                frames = []
                fps = float(container.streams.video[0].average_rate) or 30.0
                step = max(1, int(fps / FRAME_SAMPLE_RATE))
                for i, frame in enumerate(container.decode(video=0)):
                    if i % step == 0:
                        frames.append(frame.to_image().convert("RGB"))
                        if len(frames) >= FRAME_LIMIT:
                            break
                container.close()

                if not frames:
                    results[p] = None
                    continue

                emb_list = []
                all_faces = []
                for batch in chunked(frames, 16):
                    inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=USE_FP16):
                            feats = self.model.get_image_features(**inputs)
                    for img in batch:
                        fv, _ = self._detect_faces(img)
                        all_faces.extend(fv)
                    emb_list.append(feats.cpu().numpy())

                face_vecs = deduplicate_faces(all_faces)
                video_vec = l2_normalize(np.vstack(emb_list).mean(axis=0))

                results[p] = {
                    "type": "video",
                    "path": p,
                    "vector": video_vec.tolist(),
                    "face_count": len(face_vecs),
                    "face_vecs": [vec.tolist() for vec in face_vecs]
                }
            except Exception as e:
                exception(f"Video {p} failed: {e}")
                results[p] = None
        return results

    def _embed_audios(self, paths):
        results = {}
        for p in paths:
            try:
                y, sr = librosa.load(p, sr=AUDIO_SR, mono=True)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
                S_db = librosa.power_to_db(S, ref=np.max)
                norm_val = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
                img = Image.fromarray(np.uint8(norm_val * 255)).convert("RGB").resize((224, 224))
                inputs = self.processor(images=img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=USE_FP16):
                        feats = self.model.get_image_features(**inputs)
                results[p] = {"type": "audio", "vector": l2_normalize(feats.cpu().numpy()[0]).tolist()}
            except Exception as e:
                exception(f"Audio {p} failed: {e}")
                results[p] = None
        return results

    def _detect_faces(self, img):
        arr = np.array(img)
        locs = face_recognition.face_locations(arr)
        debug(f'图片的人脸位置信息：{locs}')
        encodings = face_recognition.face_encodings(arr, known_face_locations=locs)
        vecs = [l2_normalize(np.array(e)) for e in encodings] if encodings else []
        debug(f'图片的人脸向量是：{vecs}')
        return vecs, []

# ------------------- API 路由（完全模仿 m2m） -------------------
async def embed(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')

    se = ServerEnv()
    engine = se.engine

    # 从 params_kw 获取参数
    inputs = getattr(params_kw, 'inputs', None)
    if not inputs or not isinstance(inputs, list):
        raise Exception("`inputs` must be a non-empty list")

    # 调用嵌入
    raw_result = engine.embed_batch(inputs)

    # 构建标准响应
    ret = {
        "data": raw_result,
        "object": "embedding.result",
        "model": engine.model_name
    }
    return ret

async def docs(request, *args, **kw):
    return helptext

# ------------------- 服务初始化 -------------------
def init():
    rf = RegisterFunction()
    rf.register('embed', embed)
    rf.register('docs', docs)
    debug("Registered: POST /v1/embed")

# ------------------- 服务启动 -------------------
def main():
    parser = argparse.ArgumentParser(prog="CLIP Embedding Service")
    parser.add_argument('model_path', nargs='?', help="CLIP model path")
    parser.add_argument('-p', '--port', type=int, default=8883)
    parser.add_argument('-w', '--workdir', default=os.getcwd())
    args = parser.parse_args()

    config = getConfig()
    model_name = args.model_path or config.get("model_name") or CLIP_MODEL_NAME

    se = ServerEnv()
    se.engine = MM_Embedding(model_name)

    debug(f"Starting embedding service on port {args.port}")
    webserver(init, args.workdir, args.port)

if __name__ == '__main__':
    main()