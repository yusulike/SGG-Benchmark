"""NMS import shim for YOLO backbones.

Newer ultralytics (>= ~8.3.2xx) provides `ultralytics.utils.nms.non_max_suppression`
with the `return_idxs` argument this codebase relies on.  Older versions
(including the 8.3.63-era yolov12 fork) lack it, so a vendored copy of the
8.3.x implementation, extended to track kept flat anchor indices, is used as
a fallback.  Both return `(output, keepi)` where `keepi` indexes into the
original anchor axis of the prediction tensor (used downstream as `feat_idx`).
"""

try:
    from ultralytics.utils.nms import non_max_suppression  # noqa: F401
except ImportError:
    import time

    import torch
    from ultralytics.utils.ops import xywh2xyxy

    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
        return_idxs=False,
    ):
        """Perform NMS on YOLO predictions and optionally return kept anchor indices.

        Args:
            prediction (torch.Tensor): Predictions in BCN format, e.g. shape (1, 84, 6300).
            conf_thres (float): Confidence threshold.
            iou_thres (float): IoU threshold for NMS.
            classes (List[int] | None): Class indices to keep; None keeps all.
            agnostic (bool): Class-agnostic NMS.
            multi_label (bool): Keep one box per (box, class) pair above threshold.
            max_det (int): Maximum detections per image.
            nc (int): Number of classes; inferred from `prediction` when 0.
            max_nms (int): Maximum candidates fed to torchvision NMS.
            max_wh (float): Class-offset used for batched (per-class) NMS.
            return_idxs (bool): Also return the flat anchor index of each kept box.

        Returns:
            output (List[torch.Tensor]): Per-image tensors (n, 6): xyxy, conf, cls.
            idxs (List[torch.Tensor]): Only when return_idxs — per-image flat anchor
                indices (into the original N-axis of `prediction`) of kept boxes.
        """
        import torchvision  # scope import like ultralytics does

        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"
        if isinstance(prediction, (list, tuple)):  # YOLOv8 validation output: (inference_out, loss_out)
            prediction = prediction[0]
        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        device = prediction.device
        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        mi = 4 + nc  # mask start index (masks unsupported here)
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidate anchors

        time_limit = 2.0 + 0.05 * bs  # seconds to quit after
        multi_label &= nc > 1

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        t = time.time()
        output = [torch.zeros((0, 6), device=device)] * bs
        idxs = [torch.zeros((0,), dtype=torch.long, device=device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            anchor_ids = torch.arange(x.shape[0], device=device)  # flat anchor indices
            x = x[xc[xi]]  # confidence
            anchor_ids = anchor_ids[xc[xi]]

            if not x.shape[0]:
                continue

            box, cls = x.split((4, nc), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
                anchor_ids = anchor_ids[i]
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                keep = conf.view(-1) > conf_thres
                x = torch.cat((box, conf, j.float()), 1)[keep]
                anchor_ids = anchor_ids[keep]

            if classes is not None:
                keep = (x[:, 5:6] == classes).any(1)
                x = x[keep]
                anchor_ids = anchor_ids[keep]

            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                order = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence
                x = x[order]
                anchor_ids = anchor_ids[order]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, x[:, 4], iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
            idxs[xi] = anchor_ids[i]

            if (time.time() - t) > time_limit:  # stop early if too slow
                print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break

        return (output, idxs) if return_idxs else output
