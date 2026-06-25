import torch
import numpy as np
from alphaction.structures.bounding_box import BoxList
from timm.models import create_model
import modeling_finetune # Somehow this import is necessary to register the model in timm

# Model loading
# Obdet

# ActionRecognition
def load_model(
    model_name: str = "vit_small_patch16_224",
    checkpoint: str = "pretrained/checkpoint.pth",
    num_classes: int = 80,
    num_frames: int = 16,
    tubelet_size: int = 2,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load VideoMAE model with pretrained checkpoint."""
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.001,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    state_dict = state_dict.get("module", state_dict)  # handle DataParallel

    # Handle backbone./encoder. prefixes from pretrained checkpoints
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k[9:]] = v
        elif k.startswith("encoder."):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v

    # Load weights (skip head if shape mismatch)
    model_state = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in new_state_dict and new_state_dict[k].shape != model_state[k].shape:
            del new_state_dict[k]

    # Interpolate position embedding if needed
    if "pos_embed" in new_state_dict:
        pos_embed_ckpt = new_state_dict["pos_embed"]
        num_patches_model = model.patch_embed.num_patches
        num_extra = model.pos_embed.shape[-2] - num_patches_model
        orig_size = int(((pos_embed_ckpt.shape[-2] - num_extra) //
                         (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        new_size = int((num_patches_model // (num_frames // tubelet_size)) ** 0.5)

        if orig_size != new_size:
            embedding_size = pos_embed_ckpt.shape[-1]
            extra = pos_embed_ckpt[:, :num_extra]
            pos_tokens = pos_embed_ckpt[:, num_extra:].reshape(
                -1, num_frames // tubelet_size, orig_size, orig_size, embedding_size
            ).reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1, num_frames // tubelet_size, new_size, new_size, embedding_size
            ).flatten(1, 3)
            new_state_dict["pos_embed"] = torch.cat((extra, pos_tokens), dim=1)

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# Inferrence
# Preprocessing Function
def transform(frames: np.ndarray, boxes, num_frames, sample_rate, min_size, input_size, mean, std, device):
    import cv2
    T, H, W, C = frames.shape

    h, w = frames.shape[1:3]

    boxes_tensor = torch.as_tensor(
        np.stack(boxes),
        dtype=torch.float32
    ).reshape(-1, 4)

    boxes = BoxList(
        boxes_tensor,
        (w, h),
        mode="xyxy"
    )

    boxes.add_field(
        "scores",
        torch.ones(len(boxes), 1)
    )

    # Temporal crop: sample num_frames at sample_rate
    frame_span = num_frames * sample_rate
    if T > frame_span:
        start = max(0, (T - frame_span) // 2)
    else:
        start = 0
    idx = np.arange(start, min(start + frame_span, T), sample_rate)
    idx = np.clip(idx, 0, T - 1)
    # If insufficient frames, pad with repeats
    if len(idx) < num_frames:
        idx = np.pad(idx, (0, num_frames - len(idx)), mode="edge")
    idx = idx[: num_frames]
    frames = frames[idx]

    # Resize: shorter side = min_size, keep aspect ratio
    h, w = frames.shape[1], frames.shape[2]
    if w < h:
        new_w = min_size
        new_h = int(h * min_size / w)
    else:
        new_h = min_size
        new_w = int(w * min_size / h)

    resized = np.zeros((num_frames, new_h, new_w, 3), dtype=np.uint8)
    for i in range(num_frames):
        cv2.resize(frames[i], (new_w, new_h), resized[i])

    if boxes is not None:
        boxes = boxes.resize((new_w, new_h))

    # Center crop to input_size x input_size
    h_c, w_c = new_h, new_w
    if h_c > input_size:
        y1 = (h_c - input_size) // 2
        y2 = y1 + input_size
        resized = resized[:, y1:y2, :, :]
        if boxes is not None:
            boxes = boxes.crop([0, y1, w_c, y2])
    if w_c > input_size:
        x1 = (w_c - input_size) // 2
        x2 = x1 + input_size
        resized = resized[:, :, x1:x2, :]
        if boxes is not None:
            boxes = boxes.crop([x1, 0, x2, resized.shape[1]])

    # To tensor: [T, H, W, C] -> [C, T, H, W]
    clip = torch.from_numpy(resized.transpose(3, 0, 1, 2).astype(np.float32))

    # Normalize: (x - mean) / std
    for t, m, s in zip(clip, mean, std):
        t.sub_(m).div_(s)

    clip = clip.unsqueeze(0).to(device)
    boxes = boxes.to(device)

    return clip, boxes

# Inference Function
@torch.no_grad()
def predict_logits(model, clip, boxlist):
    features = model.forward_features(
        clip,
        [boxlist]
    )

    logits = model.head(features)

    return features, logits

# Using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = load_model(model_name="vit_small_patch16_224", checkpoint="/home/vixmoai/Desktop/VIXMO/Akmal/VideoMAE-Action-Detection/finetune-small/checkpoint-29.pth", num_frames=16, num_classes=80)

# Prepare the input frames and boxes
window_frames = np.random.randint(0, 255, (32, 1080, 1920, 3), dtype=np.uint8)  # Simulated selected frames
person_boxes = np.array([[951.4639, 253.15569, 1155.4781, 741.2252], [733.1867, 213.12463, 904.68396, 597.9973]]) # 2 persons

# Preprocess input frames and boxes
clip, boxlist = transform(window_frames, person_boxes, num_frames=16, sample_rate=2, min_size=256, input_size=224, mean=[122.7717, 115.9465, 102.9801], std=[57.375, 57.375, 57.375], device=device)

# Predict logits
print("Clip shape:", clip.shape) # torch.Size([1, 3, 16, 224, 224])
print("BoxList shape:", boxlist.bbox.shape) # torch.Size([2, 4])
features, logits = predict_logits(model, clip, boxlist)
print("Features shape:", features.shape) # torch.Size([2, 384])
print("Logits shape:", logits.shape) # torch.Size([2, 80])
