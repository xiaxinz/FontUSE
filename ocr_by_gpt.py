import cv2
import os
import numpy as np
import json
import base64
from openai import OpenAI


OPENAI_API_KEY = 'YOUR_API_KEY_HERE'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Default model used if caller does not specify one.
DEFAULT_MODEL = "gpt-4o-mini"

imgDir = "testdata-3/resized-imgs"
bboxDir = "testdata-3/bbox"
ocrDir = "testdata-3/ocr"

def gptOCR(messages, model=None):
    """
    Call GPT-based OCR. If model is None, use DEFAULT_MODEL.
    """
    use_model = model or DEFAULT_MODEL
    response = client.chat.completions.create(
        model=use_model,
        messages=messages,
    )
    return response


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def cv2_base64(cv2_image, img_format):
    encoded = cv2.imencode(f'.{img_format}', cv2_image)[1].tobytes()
    base64_str = base64.b64encode(encoded).decode('utf-8')
    return base64_str

# process one image & its bbox JSON, return list of OCR entries and list of non-roman filenames
def ocr_bboxes_for_image(image_path, jsonFile, imgFormat, model=None):
    """
    Process one image and its bbox JSON.
    model: optional, select which GPT model to use (string). If None, DEFAULT_MODEL is used.
    Returns: (results_list, nonRoman_list)
    """
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    with open(jsonFile, 'r') as f:
        try:
            bboxData = json.load(f)
        except Exception:
            bboxData = []

    # If no bboxes, return a single "empty" entry to preserve alignment with previous behavior
    if not bboxData:
        ocrData = {
            "polygon": [
                [0.01, 0.01],
                [0.99, 0.01],
                [0.99, 0.99],
                [0.01, 0.99]],
            "language": "None",
            "text": ""
        }
        return [ocrData], []

    results = []
    nonRomanLocal = []
    for dic in bboxData:
        try:
            bboxes = np.array(dic['polygon'])
        except Exception:
            # add an empty result for this bbox and continue
            ocrData = {
                "polygon": [
                    [0.01, 0.01],
                    [0.99, 0.01],
                    [0.99, 0.99],
                    [0.01, 0.99]],
                "language": "None",
                "text": ""
            }
            results.append(ocrData)
            continue

        # bbox(relative coordinates) to mask
        imgSizeArray = np.array([[img_w, 0],
                                 [0, img_h]])
        bboxes_px = np.matmul(bboxes, imgSizeArray)

        # apply mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        polygon = np.array([bboxes_px[0], bboxes_px[1], bboxes_px[2], bboxes_px[3]], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        image_masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

        # OCR by GPT
        base64_image = cv2_base64(image_masked, imgFormat)
        text_prompt = (
            'For the given image, please carefully analyze the text in the image and OCR. '
            'Only answer the extracted text. OCR should be case-sensitive. '
            'Only extact one word. '
            'If you cannot detect any text, answer "-" '
            'If you detected non-roman characters, answer "#" '
            'it is very hard to read and you must run multiple OCR carefully to get the perfect result we are looking for.'
        )
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }]
        # pass model through to the GPT call
        res = gptOCR(messages, model=model)
        text = str(res.choices[0].message.content)
        if text == "#":
            nonRomanLocal.append(os.path.basename(image_path))

        dic_copy = dic.copy()
        dic_copy.update({"text": text})
        results.append(dic_copy)

    return results, nonRomanLocal

# process a directory (callable from other programs)
def process_directory(imgDir=imgDir, bboxDir=bboxDir, ocrDir=ocrDir, model=None):
    """
    Batch process a directory of images. model: optional GPT model name.
    """
    length = len(os.listdir(imgDir))
    m = 0
    nonRomanList = []
    for image in os.listdir(imgDir):
        m += 1
        print(f'Processing {m}/{length}: {image}')
        if os.path.isdir(os.path.join(imgDir, image)):
            continue
        imgName, imgFormat = os.path.splitext(image)
        imgName = imgName
        imgFormat = imgFormat.lstrip('.')
        jsonFile = os.path.join(bboxDir, f'{imgName}_bbox.json')
        ocrFile = os.path.join(ocrDir, f'{imgName}_ocr.json')

        if os.path.exists(ocrFile):
            continue

        image_path = os.path.join(imgDir, image)
        if not os.path.exists(jsonFile):
            # write empty list to output to keep consistent format
            with open(ocrFile, 'w') as f_new:
                json.dump([], f_new, indent=4)
            continue

        results, local_non = ocr_bboxes_for_image(image_path, jsonFile, imgFormat, model=model)
        # write the entire list at once to ensure valid JSON array format
        with open(ocrFile, 'w') as f_new:
            json.dump(results, f_new, indent=4)
        nonRomanList.extend(local_non)

    # write aggregated non-roman list
    with open("non_roman_list.json", "w") as lf:
        json.dump(nonRomanList, lf)

# allow importing and calling process_directory() from other modules, and still runnable as script
if __name__ == "__main__":
    process_directory(imgDir, bboxDir, ocrDir)

# Simple single-image wrapper for convenience
def ocr_single_image(image_path, bbox_json, img_format, model=None):
    """
    Convenience wrapper to OCR a single image. Returns (results, nonRoman_list).
    model: optional GPT model name.
    """
    return ocr_bboxes_for_image(image_path, bbox_json, img_format, model=model)

__all__ = ["process_directory", "ocr_bboxes_for_image", "ocr_single_image"]