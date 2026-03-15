'''
Detect text bounding boxes in images using Hi-SAM
'''

import json
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
import glob
from tqdm import tqdm
from PIL import Image
import random
from shapely.geometry import Polygon
import pyclipper
import datetime
import warnings
warnings.filterwarnings("ignore")

# python text_bbox_detection.py --checkpoint pretrained_checkpoints/word_detection_totaltext.pth --model-type vit_h --input [path_to_image].jpg --output [output_folder] or [output_image_name] --dataset totaltext
# python text_bbox_detection.py --checkpoint pretrained_checkpoint/word_detection_totaltext.pth --model-type vit_h --input /[path_to_image_folder] --output [output_folder] --dataset totaltext

def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image or folder.")
    parser.add_argument("--output", type=str, default='./hi_sam_output',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--model-type", type=str, default="vit_h",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--dataset", type=str, default='totaltext',
                        help="'totaltext' or 'ctw1500', or 'ic15'.")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--visualize_bbox", action='store_true')
    parser.add_argument("--zero_shot", action='store_true')
    parser.add_argument("--mask_aug", default=5, type=int,
                        help="The number of mask region augmentation.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[512, 512], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)
    return parser.parse_args()


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def polygon2rbox(polygon, image_height, image_width):
    rect = cv2.minAreaRect(polygon)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1, 2)
    return pts


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color if color is not None else np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_masks(masks, filename, image, visualize=False):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)

    latin_data = []
    img_h, img_w = image.shape[:2]

    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        # contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for cont in contours:
        #     epsilon = 0.002 * cv2.arcLength(cont, True)
        #     approx = cv2.approxPolyDP(cont, epsilon, True)
        #     pts = approx.reshape((-1, 2))
        #     if pts.shape[0] < 4:
        #         continue
        #     pts = pts.astype(np.int32)
        #     mask = cv2.fillPoly(np.zeros(mask.shape), [pts], 1)

        maskImg = Image.fromarray(mask * 255)

        # maskImg.save(f'{filename.split(".")[0]}-{i}.jpg')

        # print(polygon2rbox((mask * 255), mask.shape[0], mask.shape[1]))
        # maskCV = np.int8(mask * 255)
        # gray = cv2.cvtColor((mask * 255), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold((mask * 255), 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_size = img_w * img_h

        for cont in contours:
            # calculate minimum area rectangle
            rect = cv2.minAreaRect(cont)
            # cv2.boxPoints will get you the 4 points of the rectangle
            box = cv2.boxPoints(rect)
            # ensure the box is in the right order (clockwise)
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            # print(box)
            # left-top, right-top, right-bottom, left-bottom

            # relative coordinates
            x1 = float(box[0][0])
            y1 = float(box[0][1])
            x2 = float(box[1][0])
            y2 = float(box[1][1])
            x3 = float(box[2][0])
            y3 = float(box[2][1])
            x4 = float(box[3][0])
            y4 = float(box[3][1])

            # ignore small bbox
            bbox_size_squared = ((x1-x2)**2 + (y1-y2)**2) * ((x3-x2)**2+(y3-y2)**2)
            # print(pow(bbox_size_squared, 0.5))
            if pow(bbox_size_squared, 0.5) < (img_size * 1e-2):
                continue

            # write relative coordinates
            latin_data.append({"polygon": [
                    [x1/img_w, y1/img_h],
                    [x2/img_w, y2/img_h],
                    [x3/img_w, y3/img_h],
                    [x4/img_w, y4/img_h],
                ],
                "language": "latin", })

            # draw bbox on image
            if visualize:
                box = box.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [box], True, (0, 255, 0), thickness=3)

    if visualize:
        cv2img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{filename.split(".")[0]}-bbox.jpg', cv2img)
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.close()
    return latin_data



if __name__ == '__main__':
    args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")
    amg = AutoMaskGenerator(hisam)

    mask_aug = args.mask_aug
    visualize_bbox = args.visualize_bbox

    if args.dataset == 'totaltext':
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel
            score_thresh = 0.3
            unclip_ratio = 1.5
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == 'ctw1500':
        if args.zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm(args.input):
        img_id = os.path.basename(path).split('.')[0]

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = os.path.basename(path).split('.')[0] + '.png'
            if visualize_bbox:
                os.makedirs(os.path.join(args.output, "bbox_show/"), exist_ok=True)
                bbox_out_filename = os.path.join(args.output, "bbox_show/", img_name)
            json_out_filename = os.path.join(args.output, "bbox/", img_name)
            # skip if result file already exists
            if os.path.exists(f"{json_out_filename.split('.')[0]}_bbox.json"):
                print("skip", json_out_filename)
                continue
        else:
            assert len(args.input) == 1
            if visualize_bbox:
                bbox_out_filename = args.output
            json_out_filename = os.path.splitext(args.output)[0] + '_bbox.json'

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
        img_h, img_w = image.shape[:2]

        amg.set_image(image)
        masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=fg_points_num,
            batch_points_num=min(fg_points_num, 100),
            score_thresh=score_thresh,
            nms_thresh=score_thresh,
            zero_shot=args.zero_shot,
            dataset=args.dataset
        )

        if masks is not None:
            print('Inference done. Start plotting masks.')
            latin_data = save_masks(masks, bbox_out_filename, image)

        else:
            print('No prediction. Mask all.')
            latin_data = {"polygon": [
                    [0.01, 0.01],
                    [0.99, 0.01],
                    [0.99, 0.99],
                    [0.01, 0.99],
                ],
                "language": "None", }

        jsonOutputDir = f"{json_out_filename.split('.')[0]}_bbox.json"
        if os.path.exists(jsonOutputDir):
            with open(jsonOutputDir, "w") as f:
                json.dump(latin_data, f, indent=4)
        else:
            with open(jsonOutputDir, "a") as f:
                json.dump(latin_data, f, indent=4)