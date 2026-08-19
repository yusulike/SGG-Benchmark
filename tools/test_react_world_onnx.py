import cv2
import numpy as np
import onnxruntime as ort


ONNX_PATH = "checkpoints/SAFETY/react_world/react_world.onnx"
IMAGE_PATH = r"E:\myWork\myYolo\test.jpg"


def letterbox(image, size=640):
    h, w = image.shape[:2]

    r = min(size / h, size / w)

    nw = int(round(w * r))
    nh = int(round(h * r))

    image = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR,
    )

    dw = size - nw
    dh = size - nh

    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return image


def preprocess(image):
    image = letterbox(image, 640)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    image = np.ascontiguousarray(image)

    # uint8 -> float32
    image = image.astype(np.float32) / 255.0

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    return image

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", required=True)
    parser.add_argument(
        "--source",
        required=True,
        help="Image path"
    )
    parser.add_argument(
        "--provider",
        default="CUDAExecutionProvider",
        choices=[
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    args = parser.parse_args()

    print("Loading ONNX...")

    session = ort.InferenceSession(
        args.onnx,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Providers:")
    print(session.get_providers())

    print()

    for x in session.get_inputs():
        print(
            "INPUT:",
            x.name,
            x.shape,
            x.type,
        )

    for x in session.get_outputs():
        print(
            "OUTPUT:",
            x.name,
            x.shape,
            x.type,
        )

    image = cv2.imread(args.source)

    if image is None:
        raise RuntimeError(
            f"Cannot load image: {args.source}"
        )

    print()


    # Input tensor
    input_tensor = preprocess(image)

    print("Original image:", image.shape)
    print(
        "Input tensor:",
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.min(),
        input_tensor.max()
    )

    # ONNX input name
    input_name = session.get_inputs()[0].name

    # ONNX inference
    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    # Outputs
    boxes = outputs[0]
    rels = outputs[1]

    print("\n================================")
    print("RAW ONNX OUTPUT")
    print("================================")

    print("boxes shape:", boxes.shape)
    print("boxes dtype:", boxes.dtype)
    print("boxes:")
    print(boxes)

    print("\nrels shape:", rels.shape)
    print("rels dtype:", rels.dtype)
    print("rels:")
    print(rels)

    print()
    print("================================")
    print("RAW OUTPUT")
    print("================================")

    print(
        "boxes:",
        boxes.shape,
        boxes.dtype,
    )

    print(
        "rels:",
        rels.shape,
        rels.dtype,
    )

    np.set_printoptions(
        precision=5,
        suppress=True,
        linewidth=200,
    )

    print()
    print("BOXES:")
    print(boxes)

    print()
    print("RELS:")
    print(rels)


if __name__ == "__main__":
    main()