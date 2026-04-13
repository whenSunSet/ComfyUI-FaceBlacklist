import os
import cv2
import torch
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize

# 获取当前节点所在绝对目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BLACKLIST_DB_DIR = os.path.join(CURRENT_DIR, "blacklist_db")
CACHE_FILE = os.path.join(CURRENT_DIR, "blacklist.npy")


class FaceRiskSystemNode:
    def __init__(self):
        """
        初始化风控系统 (随 ComfyUI 启动时加载一次)
        """
        print("[FaceRiskSystem] 初始化模型...")
        # 默认使用 CPU。如果你的环境配好了 onnxruntime-gpu，可以改成 ['CUDAExecutionProvider']
        providers = ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.blacklist_vectors = None
        self.blacklist_files = []

        # 启动时自动构建或加载黑名单库
        self.build_blacklist()

    def _get_embedding(self, img_array):
        """提取最大人脸的特征"""
        faces = self.app.get(img_array)
        if not faces:
            return None, None
        target_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
        return target_face.embedding, target_face.bbox

    def build_blacklist(self):
        """构建违禁库"""
        # 1. 尝试从缓存读取
        if os.path.exists(CACHE_FILE):
            try:
                data = np.load(CACHE_FILE, allow_pickle=True).item()
                self.blacklist_vectors = data['vectors']
                self.blacklist_files = data['filenames']
                print(f"[FaceRiskSystem] ✅ 成功加载 {len(self.blacklist_files)} 个黑名单身份。")
                return
            except Exception as e:
                print(f"[FaceRiskSystem] ❌ 缓存读取失败，将重新构建: {e}")

        # 2. 如果没有缓存，从 blacklist_db 文件夹读取图片构建
        print(f"[FaceRiskSystem] Building blacklist from folder: {BLACKLIST_DB_DIR} ...")
        vectors_list = []
        files_list = []

        if not os.path.exists(BLACKLIST_DB_DIR):
            os.makedirs(BLACKLIST_DB_DIR)
            print(f"[FaceRiskSystem] ⚠️ 文件夹 '{BLACKLIST_DB_DIR}' 已创建，请把黑名单图片放进去。")
            return

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for filename in os.listdir(BLACKLIST_DB_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            path = os.path.join(BLACKLIST_DB_DIR, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            embedding, _ = self._get_embedding(img)
            if embedding is not None:
                vectors_list.append(embedding)
                files_list.append(filename)
                print(f"  -> Added: {filename}")
            else:
                print(f"  -> Skipped (No face): {filename}")

        if vectors_list:
            # 转换为矩阵并归一化
            self.blacklist_vectors = np.array(vectors_list)
            self.blacklist_vectors = normalize(self.blacklist_vectors)
            self.blacklist_files = files_list

            # 保存缓存 (保存为字典)
            np.save(CACHE_FILE, {'vectors': self.blacklist_vectors, 'filenames': self.blacklist_files})
            print(f"[FaceRiskSystem] ✅ 构建完成，保存了 {len(vectors_list)} 个人脸特征到缓存。")
        else:
            print("[FaceRiskSystem] ⚠️ 违禁库文件夹中未找到有效人脸。")

    # ================= ComfyUI 节点定义 =================

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI 传入的图像张量
                "threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    # 输出：画了框的图片，是否违规的布尔值，详细信息文本
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("annotated_image", "is_blacklisted", "info")
    FUNCTION = "check_image_comfy"
    CATEGORY = "Face Detection"

    def check_image_comfy(self, image, threshold):
        # image 的 shape 通常是 [Batch, Height, Width, Channels] (RGB 0~1)
        out_images = []
        is_blacklisted_batch = False
        info_msgs = []

        for i in range(len(image)):
            # 1. ComfyUI Tensor 转换为 OpenCV BGR 格式
            img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # --- 开始你的核心风控逻辑 ---
            if self.blacklist_vectors is None or len(self.blacklist_vectors) == 0:
                out_images.append(img_bgr)
                # 黑名单为空，默认安全
                info_msgs.append("politics success")
                continue

            faces = self.app.get(img_bgr)
            if not faces:
                out_images.append(img_bgr)
                # 没检测到脸，默认安全
                info_msgs.append("politics success")
                continue

            # 初始状态设为成功
            msg = "politics success"

            for face in faces:
                target_emb = face.embedding.reshape(1, -1)
                target_emb = normalize(target_emb)

                # 矩阵运算
                similarities = np.dot(target_emb, self.blacklist_vectors.T).flatten()
                best_match_idx = np.argmax(similarities)
                max_score = similarities[best_match_idx]

                box = face.bbox.astype(int)
                color = (0, 255, 0)  # 绿色安全

                if max_score > threshold:
                    is_blacklisted_batch = True
                    # 只要有一张脸超过阈值，就判定为失败
                    msg = "politics fail"

                    color = (0, 0, 255)  # 红色违规
                    # 依然在图片上画出得分，方便你人工排查 (不需要的话可以删掉这句)
                    cv2.putText(img_bgr, f"RISK: {max_score:.2f}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)

            info_msgs.append(msg)
            out_images.append(img_bgr)

        # 2. 将画好框的 OpenCV 图像转回 ComfyUI Tensor 格式
        out_tensors = []
        for img_bgr in out_images:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
            out_tensors.append(tensor)

        # 拼接 Batch
        final_image_tensor = torch.stack(out_tensors)

        # 如果 batch 大于 1，用换行符拼接。一般出单图时就只有一行。
        final_info = "\n".join(info_msgs)

        return (final_image_tensor, is_blacklisted_batch, final_info)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "FaceRiskSystemNode": FaceRiskSystemNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceRiskSystemNode": "🛡️ Face Risk System"
}
