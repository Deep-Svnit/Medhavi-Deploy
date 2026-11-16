import argparse
import json
import os
import sys
import time
import traceback
from io import BytesIO
from PIL import Image
import base64

# The serialized pipeline was created with these classes defined in the
# `phytoplankton_pipeline` module. Importing it before pickle.load restores
# the class registry in a clean Render process as well as locally.
from phytoplankton_pipeline import (
    PhytoplanktonFullPipeline,
    PhytoplanktonPreprocessingPipeline,
    MobileNetV2Reg,
    ResNet18Reg,
)

# Attempt heavy imports lazily; if they fail, return error JSON

def load_pickle(path):
    import pickle
    import torch
    original_torch_load = torch.load

    # The training artifact contains CUDA tensors. Render's CPU service must
    # remap those storages while unpickling; the pipeline itself already
    # defaults to a CPU device for prediction.
    def cpu_torch_load(*args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return original_torch_load(*args, **kwargs)

    torch.load = cpu_torch_load
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        finally:
            torch.load = original_torch_load

def pil_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def main():
    parser = argparse.ArgumentParser(description='Run phytoplankton full pipeline inference on one image.')
    parser.add_argument('--pipeline', required=True, help='Path to phytoplankton_full_pipeline.pkl')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--topk', type=int, default=10, help='Top K classes to include')
    parser.add_argument('--threshold', type=float, default=None, help='Optional override threshold')
    parser.add_argument('--return-image', action='store_true', help='Return processed image (base64)')
    args = parser.parse_args()

    start_time = time.time()
    try:
        payload = load_pickle(args.pipeline)
        full_pipeline = payload.get('pipeline') if isinstance(payload, dict) else payload
        # The artifact was trained with CUDA, but Render's default service is
        # CPU-only. Ensure both the pipeline and every model use CPU tensors.
        import torch
        full_pipeline.device = torch.device('cpu')
        for model in getattr(full_pipeline, 'models', {}).values():
            model.to('cpu')
        if args.threshold is not None and hasattr(full_pipeline, 'threshold'):
            full_pipeline.threshold = args.threshold

        if not os.path.exists(args.image):
            raise FileNotFoundError(f'Image not found: {args.image}')

        result = full_pipeline.predict(args.image)
        if not isinstance(result, dict):
            raise TypeError('Pipeline returned a non-dictionary result')
        counts = result.get('counts', {})
        presence = result.get('presence', {})
        top_items = result.get('top10', [])[:args.topk]
        if not top_items:
            top_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:args.topk]
        rois = result.get('rois', [])

        processed_image_b64 = None
        if args.return_image and 'processed_image' in result:
            buf = BytesIO()
            try:
                result['processed_image'].save(buf, format='PNG')
                processed_image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception:
                processed_image_b64 = None

        duration = time.time() - start_time
        out = {
            'success': True,
            'timing': round(duration, 3),
            'top': [
                {'class': item[0], 'value': float(item[1])}
                if isinstance(item, (tuple, list)) else item
                for item in top_items
            ],
            'counts': counts,
            'presence': presence,
            'roi_count': len(rois),
            'processed_image_base64': processed_image_b64,
        }
        print(json.dumps(out))
    except Exception as e:
        err = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(limit=3)
        }
        print(json.dumps(err))
        sys.exit(1)

if __name__ == '__main__':
    main()
