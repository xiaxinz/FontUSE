import os
import sys
import argparse
import subprocess

from ocr_by_gpt import process_directory as ocr_process_directory
from usecase_gen import process_images as caption_process_images


def run_bbox_detection(
    images_dir,
    base_dir,
    checkpoint,
    model_type="vit_h",
    device="cuda",
    dataset="totaltext",
    zero_shot=False,
    mask_aug=5,
    visualize_bbox=False,
):
    """
    Run Hi-SAM text detection script on all images in images_dir.
    This function simply wraps the original text_bbox_detection.py as a subprocess.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detector_script = os.path.join(script_dir, "text_bbox_detection.py")

    if not os.path.isfile(detector_script):
        raise FileNotFoundError(f"Cannot find text_bbox_detection.py next to this script: {detector_script}")

    cmd = [
        sys.executable,
        detector_script,
        "--input",
        images_dir,
        "--output",
        base_dir,
        "--checkpoint",
        checkpoint,
        "--model-type",
        model_type,
        "--device",
        device,
        "--dataset",
        dataset,
        "--mask_aug",
        str(mask_aug),
    ]

    if zero_shot:
        cmd.append("--zero_shot")
    if visualize_bbox:
        cmd.append("--visualize_bbox")

    print("=== Step 1: text bbox detection (Hi-SAM) ===")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ocr(
    images_dir,
    bbox_dir,
    ocr_dir,
    model=None,
):
    """
    Run GPT-based OCR on all images using pre-computed bbox JSON files.
    """
    os.makedirs(ocr_dir, exist_ok=True)
    print("=== Step 2: OCR with GPT ===")
    print(f"Images dir : {images_dir}")
    print(f"BBox dir   : {bbox_dir}")
    print(f"OCR out dir: {ocr_dir}")
    ocr_process_directory(imgDir=images_dir, bboxDir=bbox_dir, ocrDir=ocr_dir, model=model)


def run_caption(
    images_dir,
    caption_dir,
    model=None,
    split_into_three=False,
):
    """
    Run GPT-based font style / use-case caption generation on all images.
    """
    os.makedirs(caption_dir, exist_ok=True)
    print("=== Step 3: caption / use-case generation ===")
    print(f"Images dir   : {images_dir}")
    print(f"Caption out  : {caption_dir}")
    caption_process_images(
        img_dirs=[images_dir],
        output_dir=caption_dir,
        split_into_three=split_into_three,
        model=model,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline: Hi-SAM text detection -> GPT OCR -> caption/use-case generation."
    )
    parser.add_argument(
        "images_dir",
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Hi-SAM checkpoint file used by text_bbox_detection.py.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for Hi-SAM model (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--dataset",
        default="totaltext",
        choices=["totaltext", "ctw1500"],
        help="Dataset preset for Hi-SAM detection (affects thresholds).",
    )
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="Hi-SAM backbone type.",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Use zero-shot thresholds for Hi-SAM.",
    )
    parser.add_argument(
        "--visualize-bbox",
        action="store_true",
        help="Additionally save images with drawn bounding boxes.",
    )
    parser.add_argument(
        "--mask-aug",
        type=int,
        default=5,
        help="Number of mask region augmentations for Hi-SAM (mask_aug).",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="Optional GPT model name for OCR (overrides ocr_by_gpt.DEFAULT_MODEL).",
    )
    parser.add_argument(
        "--caption-model",
        default=None,
        help="Optional GPT model name for caption/use-case generation (overrides usecase_gen.DEFAULT_MODEL).",
    )
    parser.add_argument(
        "--split-caption-into-three",
        action="store_true",
        help="Split usecases into up to three JSON files per image (see usecase_gen.py for details).",
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="Skip Hi-SAM step and reuse existing bbox JSON files in the 'bbox' directory.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR step and only run caption generation.",
    )

    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"images_dir does not exist or is not a directory: {images_dir}")

    base_dir = os.path.dirname(images_dir)
    bbox_dir = os.path.join(base_dir, "bbox")
    ocr_dir = os.path.join(base_dir, "ocr")
    caption_dir = os.path.join(base_dir, "caption")

    # Ensure bbox directory exists before running detection
    os.makedirs(bbox_dir, exist_ok=True)

    print("Input images  :", images_dir)
    print("BBox dir      :", bbox_dir)
    print("OCR dir       :", ocr_dir)
    print("Caption dir   :", caption_dir)
    print()

    if not args.skip_bbox:
        run_bbox_detection(
            images_dir=images_dir,
            base_dir=base_dir,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            dataset=args.dataset,
            zero_shot=args.zero_shot,
            mask_aug=args.mask_aug,
            visualize_bbox=args.visualize_bbox,
        )
    else:
        print("Skipping Step 1 (bbox detection); existing bbox JSONs will be reused.")

    if not args.skip_ocr:
        run_ocr(
            images_dir=images_dir,
            bbox_dir=bbox_dir,
            ocr_dir=ocr_dir,
            model=args.ocr_model,
        )
    else:
        print("Skipping Step 2 (OCR).")

    run_caption(
        images_dir=images_dir,
        caption_dir=caption_dir,
        model=args.caption_model,
        split_into_three=args.split_caption_into_three,
    )

    print("=== All done ===")
    print("Bounding boxes JSON : ", bbox_dir)
    print("OCR JSON files      : ", ocr_dir)
    print("Caption JSON files  : ", caption_dir)


if __name__ == "__main__":
    main()
