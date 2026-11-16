"""Phytoplankton pipeline core definitions.

Place this file in your PYTHONPATH (already in workspace root). Import BEFORE loading any
pickled objects that reference these classes (e.g. phytoplankton_full_pipeline.pkl).

Example:
    import pickle
    from phytoplankton_pipeline import PhytoplanktonFullPipeline  # ensures class is defined
    with open(r'e:/SIH AI/Pelgas Dataset/phytoplankton_full_pipeline.pkl', 'rb') as f:
        payload = pickle.load(f)
    full_pipeline = payload['pipeline']
    result = full_pipeline.predict(r'e:/SIH AI/Pelgas Dataset/deep dataset/00001.jpg')
    print(result['top10'])
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import filters, morphology, exposure, measure
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------------
# Preprocessing Utility Functions (from preprocessing_pipeline.ipynb)
# ---------------------------------------------------------------------------

def spectral_gating(img: Image.Image, threshold: float = 0.1) -> Image.Image:
    """Frequency domain noise suppression.

    Args:
        img: PIL Image (any mode)
        threshold: relative threshold for magnitude spectrum mask
    Returns:
        Denoised grayscale PIL Image
    """
    gray = np.array(img.convert('L'))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    mask = magnitude_spectrum > (threshold * magnitude_spectrum.max())
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back / (img_back.max() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(img_back)


def detect_rois(img: Image.Image, min_area: int = 100) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """Detect simple ROIs via threshold + morphology.

    Returns image with red rectangles and list of bounding boxes (minr, minc, maxr, maxc).
    """
    gray = np.array(img.convert('L'))
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh
    cleaned = morphology.remove_small_objects(binary, min_size=min_area)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3))
    labels = measure.label(cleaned)
    rois = []
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            rois.append(region.bbox)
    img_roi = img.copy()
    draw = ImageDraw.Draw(img_roi)
    for (minr, minc, maxr, maxc) in rois:
        draw.rectangle([minc, minr, maxc, maxr], outline='red', width=2)
    return img_roi, rois


def preprocess_image(img: Image.Image) -> Image.Image:
    """End-to-end preprocessing: spectral gating -> median filter -> CLAHE -> normalization."""
    img_sg = spectral_gating(img)
    img_dn = img_sg.filter(ImageFilter.MedianFilter(size=3))
    img_arr = np.array(img_dn)
    img_eq = exposure.equalize_adapthist(img_arr, clip_limit=0.03)
    img_eq = (img_eq * 255).astype(np.uint8)
    img_norm = (img_eq - np.mean(img_eq)) / (np.std(img_eq) + 1e-8)
    scaled = ((img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(scaled)


# ---------------------------------------------------------------------------
# Preprocessing Pipeline Class
# ---------------------------------------------------------------------------
class PhytoplanktonPreprocessingPipeline:
    def __init__(self, min_roi_area: int = 100, spectral_threshold: float = 0.1):
        self.min_roi_area = min_roi_area
        self.spectral_threshold = spectral_threshold

    def process(self, img: Union[str, Image.Image]):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        img_pre = preprocess_image(img)
        img_roi, rois = detect_rois(img_pre, min_area=self.min_roi_area)
        return img_roi, rois


# ---------------------------------------------------------------------------
# Lightweight CNN Regression Models (as in stacking notebook)
# ---------------------------------------------------------------------------
class MobileNetV2Reg(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.base = torchvision.models.mobilenet_v2(weights=None)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.last_channel, num_outputs)
        )

    def forward(self, x):  # type: ignore
        return self.base(x)


class ResNet18Reg(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()
        self.base = torchvision.models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_outputs)

    def forward(self, x):  # type: ignore
        return self.base(x)


# ---------------------------------------------------------------------------
# Unified Full Pipeline Wrapper (used in merged pickle)
# ---------------------------------------------------------------------------
class PhytoplanktonFullPipeline:
    def __init__(
        self,
        preprocessing_pipeline: PhytoplanktonPreprocessingPipeline,
        models: Dict[str, nn.Module],
        meta_learner: Any,
        class_columns: List[str],
        threshold: float = 0.5,
        device: torch.device | None = None
    ):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.models = models
        self.meta_learner = meta_learner
        self.class_columns = class_columns
        self.threshold = threshold
        self.device = device or torch.device('cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _prepare_image(self, img: Union[str, Image.Image]):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        processed_img, rois = self.preprocessing_pipeline.process(img)
        # Preprocessing produces a grayscale image, while the trained CNNs
        # expect three ImageNet-normalized channels.
        tensor = self.transform(processed_img.convert('RGB')).unsqueeze(0).to(self.device)
        return tensor, rois, processed_img

    def predict(self, img: Union[str, Image.Image]) -> Dict[str, Any]:
        tensor, rois, processed_img = self._prepare_image(img)
        base_preds = []
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                out = model(tensor).cpu().numpy()
            base_preds.append(out)
        import numpy as _np  # local alias to avoid confusion
        X_stack = _np.hstack(base_preds)
        y_pred = self.meta_learner.predict(X_stack)[0]
        counts = {cls: float(val) for cls, val in zip(self.class_columns, y_pred)}
        presence = {cls: int(val > self.threshold) for cls, val in zip(self.class_columns, y_pred)}
        summary = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            'counts': counts,
            'presence': presence,
            'rois': rois,
            'top10': summary,
            'processed_image': processed_img
        }


__all__ = [
    'spectral_gating',
    'detect_rois',
    'preprocess_image',
    'PhytoplanktonPreprocessingPipeline',
    'MobileNetV2Reg',
    'ResNet18Reg',
    'PhytoplanktonFullPipeline'
]
