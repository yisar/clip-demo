import io
import requests
import torch
import matplotlib.pyplot as plt
import open_clip
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = 448 # AnyUp 建议尺寸

# ===================== 1. 加载并预处理图像 =====================
url = "https://dd-static.jd.com/ddimgp/jfs/t20270412/402218/6/3083/4812860/69ad6837Fb5f43e3d/0936bb8fa04cf2c5.jpg"
try:
    img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
except:
    img = Image.new('RGB', (img_size, img_size), color='gray')

# ===================== 2. 加载 ViT-B/16 模型 =====================
# 修改这里：使用 ViT-B-16
model_name = "ViT-B-16" 
pretrained = "laion2b_s34b_b88k" # 对应的优秀权重
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()
visual = model.visual

# ===================== 3. 图像预处理 =====================
from torchvision import transforms
anyup_preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
hr_image = anyup_preprocess(img).unsqueeze(0).to(device)
clip_img = preprocess(img).unsqueeze(0).to(device)

# ===================== 4. 提取特征 (修复维度报错) =====================
with torch.no_grad():
    # 1. 卷积投影
    x = visual.conv1(clip_img) 
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) 
    
    # 2. 准备 Class Token (1, 1, 768)
    cls_token = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    
    # 3. 拼接得到完整序列 [1, 197, 768] (1+14*14)
    x = torch.cat([cls_token, x], dim=1) 
    
    # 4. 加上位置编码 (关键修复：确保维度完全匹配)
    pos_emb = visual.positional_embedding.to(x.dtype)
    if pos_emb.ndim == 2:
        pos_emb = pos_emb.unsqueeze(0)
    
    x = x + pos_emb # 现在的 x 和 pos_emb 都是 197 长度
    
    # 5. Transformer 传播
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    
    # 6. 取出空间 Tokens [1, 196, 768]
    tokens = x[:, 1:, :]

# ===================== 5. AnyUp & 可视化 =====================
B, N, C = tokens.shape
h = w = int(N ** 0.5)
lr_features = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

# AnyUp 上采样
upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
with torch.no_grad():
    hr_features = upsampler(hr_image, lr_features)

# PCA 可视化逻辑
def get_pca_rgb(feats, h_w_size):
    f = feats[0].permute(1, 2, 0).reshape(-1, feats.shape[1])
    f = f - f.mean(0)
    _, _, V = torch.pca_lowrank(f, q=3)
    proj = f @ V
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    return proj.reshape(h_w_size, h_w_size, 3).cpu().numpy()

lr_rgb = get_pca_rgb(lr_features, h)
hr_rgb = get_pca_rgb(hr_features, img_size)

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(lr_rgb); axs[0].set_title(f"CLIP B/16 ({h}x{h})"); axs[0].axis("off")
axs[1].imshow(hr_rgb); axs[1].set_title(f"AnyUp ({img_size}x{img_size})"); axs[1].axis("off")
plt.show()