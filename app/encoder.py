import timm
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "vit_small_patch14_dinov2",
    pretrained=True,
    num_classes=0,
    in_chans=1,
)
model.eval().to(device)

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """Chuyển numpy array grayscale thành tensor đã chuẩn hoá."""
    pil = Image.fromarray(img_np)
    return transform(pil)  # (1, H, W)


@torch.no_grad()
def extract_feature(img_np: np.ndarray) -> torch.Tensor:
    """Trích xuất feature vector L2-normalised cho 1 ảnh."""
    x = _to_tensor(img_np).unsqueeze(0).to(device)  # (1, 1, H, W)
    feat = model(x)                                   # (1, D)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0)                            # (D,)


@torch.no_grad()
def batch_extract_features(imgs_np: list) -> torch.Tensor:
    """Trích xuất features cho nhiều ảnh trong MỘT forward pass duy nhất.

    Trả về tensor shape (N, D) đã L2-normalised.
    """
    if not imgs_np:
        return torch.empty(0)

    batch = torch.stack([_to_tensor(img) for img in imgs_np]).to(device)  # (N, 1, H, W)
    feats = model(batch)                                                    # (N, D)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats                                                            # (N, D)