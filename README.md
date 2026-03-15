# FontUSE Annotation Pipeline

## Overview

This repository contains a small pipeline used to generate training data for typography-aware datasets.  
Given a folder of images, the code can automatically:

1. Detect text regions using Hi-SAM to get normalized bounding boxes.
2. Run OCR on each detected region with a GPT vision model.
3. Using a MLLM-model to annotate the font style and suitable use cases ("usecases") for each image.

The final outputs are JSON files that can be used to build datasets for controllable text rendering, font-style modeling, or other text-in-image research.

## Repository structure

- `text_bbox_detection.py`  
  Runs Hi-SAM on input images and writes one `<image>_bbox.json` file per image.  
  Each JSON contains a list of polygons in normalized coordinates `[0..1] x [0..1]`.

- `ocr_by_gpt.py`  
  Uses GPT (e.g. `gpt-4o-mini`) to perform OCR on each bounding box.  
  For every input image, it reads `<image>_bbox.json` and produces `<image>_ocr.json` with:
  - `polygon`: the bbox (copied from the bbox JSON),
  - `language`: a simple language tag,
  - `text`: OCR result for that region.  
  A `nonRomanList.json` file is also written to record samples that contain non-latin characters.

- `usecase_gen.py`  
  Uses a GPT model with a custom prompt (`prompts/system_prompt.txt`, `prompts/user_prompt.txt`) to describe:
  - a short caption for the image,
  - font style / design keywords,
  - a list of use cases where this font would be suitable.  
  For each image the script writes one or up to three caption JSON files depending on how many `usecases` are returned.

- `testdata-3/`  
  Three example images for testing.

- `hi_sam/`  
  Local Hi-SAM implementation used by `text_bbox_detection.py`.

## Installation

1. **Python packages**

   Install the required libraries (adjust versions as needed):

   ```bash
   pip install torch torchvision
   pip install opencv-python scikit-image matplotlib tqdm pyclipper
   pip install openai
   ```


2. **Hi-SAM checkpoint**

Download the required Hi-SAM checkpoint from the official [Hi-SAM](https://github.com/ymy-k/Hi-SAM) repository. See more details at the official page.
You will pass its path to the scripts via the --checkpoint argument.


3. **OpenAI API key**

Both `ocr_by_gpt.py` and `usecase_gen.py` call the OpenAI API via the official openai package.

Either edit these files and set `OPENAI_API_KEY = "YOUR_KEY_HERE"`, or

change the code to read the key from the `OPENAI_API_KEY` environment variable.

Make sure that the models you configure (default: `gpt-4o`) are available in your OpenAI account.

## One-shot pipeline: images → OCR + caption

A convenience script `run_full_pipeline.py` ties all stages together.

run:

```bash
python run_full_pipeline.py path/to/images \
  --checkpoint path/to/hi_sam_checkpoint.pth
```

The output folders are created next to your image folder:

`path/to/bbox/` – intermediate bounding box JSONs (<image>_bbox.json)

`path/to/ocr/` – OCR results (<image>_ocr.json)

`path/to/caption/` – caption & usecase JSONs (<image>_caption.json or _caption_1/2/3.json)


### Command-line options

`run_full_pipeline.py` supports a few useful flags:

`images_dir` (positional) – directory containing the input images.

`--checkpoint` (required) – path to the Hi-SAM checkpoint file.

`--device` – device for Hi-SAM (cuda or cpu, default: cuda).

`--dataset` – preset for detection thresholds (totaltext or ctw1500, default: totaltext).

`--model-type` – Hi-SAM backbone (vit_h, vit_l, vit_b).

`--zero-shot` – use the zero-shot threshold settings in text_bbox_detection.py.

`--visualize-bbox` – additionally save images with drawn bounding boxes.

`--mask-aug` – number of mask augmentations (mask_aug argument in text_bbox_detection.py).

`--ocr-model` – override the GPT model used for OCR (otherwise ocr_by_gpt.DEFAULT_MODEL is used).

`--caption-model` – override the GPT model used for caption/usecase generation.

`--split-caption-into-three` – enable the mode where usecases are split into up to three JSON files per image.

`--skip-bbox` – reuse existing bbox JSONs and skip Hi-SAM.

`--skip-ocr` – skip OCR step and only run caption generation.


## Data formats (summary)

- **Bounding box JSON** (`*_bbox.json`)
Either a list of entries or a single object containing at least:

polygon: list of 4 points `[[x1, y1], ..., [x4, y4]]` in normalized coordinates.

language: simple language tag indicating whether the text uses Latin characters.

- **OCR JSON** (`*_ocr.json`)
A list of entries, one per bbox:

```
{
  "polygon": [[0.11, 0.22], [0.33, 0.22], [0.33, 0.44], [0.11, 0.44]],
  "language": "Latin",
  "text": "Word1"
},
{
  "polygon": [[0.55, 0.66], [0.77, 0.66], [0.77, 0.88], [0.55, 0.88]],
  "language": "Latin",
  "text": "Word2"
}
```

- **Caption JSON** (`*_caption*.json`)
A dictionary with fields such as:

```
{
  "img_name": "IMG-NAME.jpg",
  "caption": "Suitable For: modern educational and creative projects.; Ideal Applications: **[creative] Marketing materials for workshops or conferences focused on innovation...**; Font Styles: bold; sans-serif; ...; Colors: yellow, white",
  "suitable-for": "modern educational and creative projects.",
  "usecases": [
    "[creative] Marketing materials for workshops or conferences focused on innovation.",
    "..."
    ],
  "styles": [
    "bold", 
    "sans-serif",
    "..."
    ],
  "colors": [
    "yellow",
    "white"
  ]
}
```

These annotations can be combined into a single dataset for downstream training or analysis.