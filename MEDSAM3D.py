# /// CONFIGURATION START:

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from flask import Blueprint, jsonify, request


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2_train.build_sam import build_sam2_video_predictor

bp = Blueprint('MedSAM3D', __name__)

CONFIG_FILENAME = 'sam2_hiera_s.yaml'
CHECKPOINT_PATH = os.path.join('checkpoints', 'sam2_hiera_small.pt')

# Print paths for debugging
print(f"Using CONFIG_FILENAME: {CONFIG_FILENAME}")
print(f"Using CHECKPOINT_PATH: {CHECKPOINT_PATH}")
print(f"Checking if checkpoint exists: {os.path.exists(CHECKPOINT_PATH)}")

predictor = build_sam2_video_predictor(CONFIG_FILENAME, CHECKPOINT_PATH, device=device)

# Global state holders
inference_state = None
ann_frame_idx = 0 # Start like it was frame 0, and only go forward
ann_obj_id = 1 # annotate only 1 object
# /// CONFIGURATION END


@bp.route('/medsam3d/init_state', methods=['POST'])
def init_state():
    """
    Initialize a new MEDSAM3D inference state for a directory of JPEG frames.
    POST JSON: { "video_dir": "/path/to/frames" }
    → { "message": "State initialized", "video_dir": "..." }
    """
    global inference_state
    data = request.get_json(force=True)
    video_dir = data.get('video_dir')
    if not video_dir or not os.path.isdir(video_dir):
        return jsonify(error="Invalid or missing 'video_dir'"), 400

    inference_state = predictor.init_state(video_path=video_dir)
    return jsonify(message="Inference state initialized", video_dir=video_dir), 200


@bp.route('/medsam3d/reset_state', methods=['POST'])
def reset_state():
    """
    Reset (clear) the current inference state, removing all masks, points, etc.
    POST with no body.
    → { "message": "State reset successfully" }
    """
    global inference_state
    if inference_state is None:
        return jsonify(error="No active inference_state. Call init_state first."), 400

    predictor.reset_state(inference_state)
    inference_state = None
    return jsonify(message="Inference state reset"), 200


@bp.route('/medsam3d/predict_combined', methods=['POST'])
def predict_combined():
    """
    Add points or a box to a single frame and get the mask logits.
    POST JSON:
      {
        "image": "data:image/jpeg;base64,...",
        "point_coords": [[x1, y1], [x2, y2], ...],
        "point_labels": [1, 0, ...]
      }
    → {
        "frame_idx": 0,
        "obj_ids": [1, ...],
        "masks": { "1": [[true,false,...], ...], ... }
      }
    """
    global inference_state, ann_frame_idx, ann_obj_id
    if inference_state is None:
        return jsonify(error="No active inference_state. Call init_state first."), 400

    data = request.get_json(force=True)
    image_data = data.get('image')
    if not image_data:
        return jsonify(error="Missing 'image' data URL"), 400

    # Convert data URL to numpy array (implement this helper separately)
    arr = _data_url_to_array(image_data)

    point_coords = np.array(data.get('point_coords', []))
    point_labels = np.array(data.get('point_labels', []))

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=point_coords,
        labels=point_labels,
    )

    # Build boolean masks
    masks = {
        int(obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().tolist()
        for i, obj_id in enumerate(out_obj_ids)
    }

    return jsonify(frame_idx=ann_frame_idx,
                   obj_ids=[int(x) for x in out_obj_ids],
                   masks=masks), 200


@bp.route('/medsam3d/propagate', methods=['POST'])
def propagate_video():
    """
    Propagate the current masks across the entire video sequence.
    Collect per-frame segmentation dict, then reset state.
    POST with no body.
    → { "video_segments": { "0": { "1": [[...],...], ... }, ... } }
    """
    global inference_state
    if inference_state is None:
        return jsonify(error="No active inference_state. Call init_state first."), 400

    video_segments = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        masks = {
            int(obj_id): (mask_logits[i] > 0.0).cpu().numpy().tolist()
            for i, obj_id in enumerate(obj_ids)
        }
        video_segments[int(frame_idx)] = masks

    # Automatically reset after propagation
    predictor.reset_state(inference_state)
    inference_state = None

    return jsonify(video_segments=video_segments), 200