import torch
from ultralytics.nn.tasks import DetectionModel

# from ultralytics.nn.tasks import load_checkpoint
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.patches import torch_load

try:
    from ultralytics.utils.plotting import feature_visualization
except ImportError:  # removed in ultralytics >= 8.4 — identical local copy
    from .utils import feature_visualization
from .yolo_nms import non_max_suppression
from pathlib import Path
from omegaconf import DictConfig
from ultralytics.nn.modules.head import Detect as _Detect

class YoloModel(DetectionModel):
    def __init__(self, cfg: DictConfig, ch: int = 3, nc: int | None = None, verbose: bool = True):  # model, input channels, number of classes
        yolo_cfg = cfg.model.yolo.size + '.yaml'
        if getattr(cfg, 'VERBOSE', None) in ["DEBUG", "INFO"]:
            verbose = True
        else:
            verbose = False
        super().__init__(yolo_cfg, nc=nc, verbose=True)

        # monkey patch for end2end: ultralytics' Detect.postprocess drops the
        # anchor index, which we need as feat_idx for feature lookup.  Signature
        # matches ultralytics >= 8.3.2xx: postprocess(preds, max_det, nc).
        def _patched_postprocess(self_detect, preds, max_det, nc):
            # preds: [B, N, 4+nc], boxes in xyxy (end2end decode skips xywh)
            n_anchors = preds.shape[1]
            boxes, scores = preds.split([4, nc], dim=-1)
            k = min(max_det, n_anchors * nc)
            flat_scores, flat_idx = scores.flatten(1).topk(k)  # [B, k]
            # flat_idx = anchor * nc + cls — recover the original flat anchor index
            anchor_idx = flat_idx // nc
            cls_idx = flat_idx % nc
            top_boxes = boxes.gather(dim=1, index=anchor_idx.unsqueeze(-1).repeat(1, 1, 4))
            return torch.cat(
                [top_boxes, flat_scores.unsqueeze(-1), cls_idx.float().unsqueeze(-1), anchor_idx.float().unsqueeze(-1)],
                dim=-1,
            )  # [B, k, 7]: xyxy, score, cls, anchor_idx

        # Apply only to the detection head instance
        head = self.model[-1]  # last module is always the Detect head
        if isinstance(head, _Detect) and head.end2end:
            import types
            head.postprocess = types.MethodType(_patched_postprocess, head)

        # self.features_layers = [len(self.model) - 2]
        self.conf_thres = cfg.model.backbone.nms_thresh
        self.iou_thres = cfg.model.roi_heads.nms
        self.device = cfg.model.device
        self.input_size = cfg.input.img_size  # (W, H)
        self.input_w = int(self.input_size[0])
        self.input_h = int(self.input_size[1])
        self.nc = nc
        self.max_det = cfg.model.roi_heads.detections_per_img

        if self.end2end or '11' in yolo_cfg or '26' in yolo_cfg:
            self.layers_to_extract = [16, 19, 22]
        elif '12' in yolo_cfg:
            self.layers_to_extract = [14, 17, 20] #[14, 17, 20] # [3,5,8]
        else:
            self.layers_to_extract = [15, 18, 21] #[15, 18, 21] # [4,6,9]
        
        # Freeze backbone: full, partial (freeze_at), or none
        freeze    = cfg.model.backbone.freeze
        freeze_at = getattr(cfg.model.backbone, 'freeze_at', -1) if not freeze else -1
        self._freeze_backbone(freeze, freeze_at)
    
    def _freeze_backbone(self, freeze: bool, freeze_at: int = -1):
        """Configure backbone parameter gradients.

        Three modes
        -----------
        freeze=True                   : freeze every parameter (full freeze, no grad).
        freeze=False, freeze_at >= 0  : freeze layers [0, freeze_at) by setting
                                        requires_grad=False; layers freeze_at.. are
                                        fully trainable (requires_grad=True).
                                        The whole model stays in eval() so that the
                                        YOLO detection head always outputs decoded
                                        (NMS-compatible) predictions.  Gradients
                                        still propagate through unfrozen layers via
                                        autograd --- eval/train mode only controls
                                        BN/dropout, not whether gradients flow.
        freeze=False, freeze_at < 0   : fine-tune the entire backbone.
        """
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            print("YOLO Backbone FROZEN - no gradients will be computed")
        elif freeze_at >= 0:
            for i, m in enumerate(self.model):
                for param in m.parameters():
                    param.requires_grad = (i >= freeze_at)
            # eval() is required so the detection head returns decoded predictions
            # that postprocess/NMS can consume.  Gradients still flow through
            # layers with requires_grad=True.
            self.eval()
            n_frozen = freeze_at
            n_train  = len(self.model) - freeze_at
            print(f"YOLO Backbone PARTIALLY FROZEN — "
                  f"{n_frozen} layers frozen (0-{freeze_at-1}), "
                  f"{n_train} layers trainable ({freeze_at}+)")
        else:
            for param in self.parameters():
                param.requires_grad = True
            print("YOLO Backbone WILL BE FINE-TUNED - gradients will be computed")

    # custom implementation of forward method based on
    # https://github.com/ultralytics/ultralytics/blob/3df9d278dce67eec7fdb4fddc0aab22fee62588f/ultralytics/nn/tasks.py#L122
    def forward(self, x, profile=False, visualize=False, embed=None):
        y, feature_maps = [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in original YOLOV8 implementation.
            """
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=Path(
                    './demo/test_custom/results'))
            if embed:
                if i in self.layers_to_extract:  # if current layer is one of the feature extraction layers
                    feature_maps.append(x)
                    
        if embed:
            return x, feature_maps
        else:
            return x

    def load(self, weights_path: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        # weights, _ = load_checkpoint(weights_path)
        ckpt = torch_load(weights_path, map_location=self.device)
        weights = (ckpt.get("ema") or ckpt["model"]).float() if isinstance(ckpt, dict) else ckpt

        if weights:
            super().load(weights)

    def postprocess(self, preds, image_sizes):
        """Post-processes predictions and returns a list of Results objects."""

        if self.end2end:
            preds = preds[0]               # [B, max_det, 7]
            mask = preds[..., 4] > self.conf_thres
            preds  = [p[mask[i]] for i, p in enumerate(preds)]
            # sort & cap
            preds  = [p[p[:, 4].argsort(descending=True)][:self.max_det] for p in preds]
            # column 6 is the real flat anchor index
            indices = [p[:, 6].long() for p in preds]
            preds   = [p[:, :6] for p in preds]   # drop feat_idx from box tensor
        else:
            preds, indices = non_max_suppression(
                preds,
                nc=self.nc,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det,
                return_idxs=True,
            )

        results = []
        for i, (pred, idx) in enumerate(zip(preds, indices)):
            # Get the image size for this image (H, W)
            orig_size = image_sizes[i]
            orig_h = float(orig_size[0])
            orig_w = float(orig_size[1])
            
            if pred.shape[0] == 0:
                # return an empty instance
                instance = {
                    "boxes": torch.zeros((0, 4), device=self.device).float(),
                    # lb_boxes: boxes in letterbox pixel coords (640×640 input space).
                    # All feature maps are computed on the letterboxed image, so any
                    # geometry-based lookup into P3/P4/P5 must use lb_boxes, not boxes.
                    "lb_boxes": torch.zeros((0, 4), device=self.device).float(),
                    "lb_input_size": self.input_h,
                    "lb_gain": 1.0,
                    "lb_pad_w": 0.0,
                    "lb_pad_h": 0.0,
                    "image_size": (int(orig_w), int(orig_h)),
                    "mode": "xyxy",
                    "pred_labels": torch.zeros((0,), device=self.device).long(),
                    "pred_scores": torch.zeros((0,), device=self.device).float(),
                    "labels": torch.zeros((0,), device=self.device).long(),
                    "feat_idx": torch.zeros((0,), device=self.device).long()
                }
                results.append(instance)
                continue

            boxes = pred[:, :4]
            # If we are exporting to ONNX, we return boxes in the input space [0, imgsz]
            # to allow the deployment script to handle letterbox reversal dynamically.
            # Otherwise, we scale to original image size for standard training/eval.
            if not torch.onnx.is_in_onnx_export():
                gain = min(self.input_h / orig_h, self.input_w / orig_w)
                
                # Offsets for letterbox
                pad_w = (self.input_w - orig_w * gain) / 2
                pad_h = (self.input_h - orig_h * gain) / 2
                
                offset_w = torch.round(torch.as_tensor(pad_w - 0.1, device=boxes.device))
                offset_h = torch.round(torch.as_tensor(pad_h - 0.1, device=boxes.device))

                # Save letterbox-space boxes BEFORE reversing the transform.
                # Feature maps P3/P4/P5 are always in this 640×640 letterbox space,
                # so any geometry-based lookup (union box center, bilinear sampling)
                # must use lb_boxes, not the original-image boxes below.
                lb_boxes = boxes.clamp(min=0)

                # Scale and shift boxes to original image space
                # and avoid inplace operations which can be tricky for ONNX (if it were used)
                b0 = (boxes[:, 0] - offset_w) / gain
                b1 = (boxes[:, 1] - offset_h) / gain
                b2 = (boxes[:, 2] - offset_w) / gain
                b3 = (boxes[:, 3] - offset_h) / gain
                
                # Clip boxes
                b0 = b0.clamp(0, orig_w)
                b1 = b1.clamp(0, orig_h)
                b2 = b2.clamp(0, orig_w)
                b3 = b3.clamp(0, orig_h)
                
                boxes = torch.stack([b0, b1, b2, b3], dim=1)
            else:
                # ONNX export: boxes are already in letterbox space
                gain, pad_w, pad_h = 1.0, 0.0, 0.0
                offset_w = offset_h = torch.zeros(1, device=boxes.device)
                lb_boxes = boxes.clamp(min=0)

            scores = pred[:, 4]
            labels = pred[:, 5].long()
            
            # add 1 to all labels to account for background class
            labels_plus_1 = labels + 1

            instance = {
                "boxes": boxes,
                # Letterbox-space boxes (640×640 pixel coords, same space as feature maps).
                # Use lb_boxes for all feature-map lookups (union box center, bilinear sampling,
                # box positional encoding).  Use boxes for GT IoU matching and visualization
                # on the original-resolution image.
                "lb_boxes": lb_boxes,
                "lb_input_size": self.input_h,
                "lb_gain": float(gain),
                "lb_pad_w": float(offset_w.item() if hasattr(offset_w, 'item') else offset_w),
                "lb_pad_h": float(offset_h.item() if hasattr(offset_h, 'item') else offset_h),
                "image_size": (int(orig_w), int(orig_h)),
                "mode": "xyxy",
                "pred_labels": labels_plus_1.detach().clone(),
                "pred_scores": scores,
                "labels": labels_plus_1,
                "feat_idx": idx.long()
            }

            results.append(instance)
        return results

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}