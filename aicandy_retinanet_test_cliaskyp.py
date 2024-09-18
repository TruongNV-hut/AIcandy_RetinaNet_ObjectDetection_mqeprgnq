"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from PIL import Image, ImageDraw, ImageFont
from aicandy_utils_src_obilenxc.model import ResNet, BasicBlock

# python aicandy_retinanet_test_cliaskyp.py --image_path image_test.jpg --model_path 'aicandy_output_ntroyvui/aicandy_model_retina_lgkrymnl.pth' --class_list labels.txt --output_path aicandy_output_ntroyvui/image_out.jpg

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    # print('labels: ', labels)
    return labels

# Draws a caption above the box in an image with custom font
def draw_caption(image, box, caption, font):
    b = np.array(box).astype(int)
    
    # Convert image to PIL format for font drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.text((b[0], b[1] - 20), caption, font=font, fill=(0, 0, 255, 0))  # red color text
    
    # Convert back to OpenCV format
    return np.array(pil_image)

def detect_image(image_path, model_path, class_list, output_path):
    labels = load_labels(class_list)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ResNet(len(labels), BasicBlock, [2, 2, 2, 2])  # Initialize the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights

    model = model.to(device)
    model.training = False
    model.eval()

    # Load the Arial font
    font_path = 'aicandy_utils_src_obilenxc/arial.ttf' 
    font = ImageFont.truetype(font_path, 16) 
    print(image_path)
    image = cv2.imread(image_path)
    image_orig = image.copy()

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():
        image = torch.from_numpy(image).to(device)
        
        st = time.time()
        print(image.shape, image_orig.shape, scale)
        
        scores, classification, transformed_anchors = model(image.float())
        
        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            print(bbox, classification.shape)
            score = scores[j]
            caption = '{} {:.3f}'.format(label_name, score)

            image_orig = draw_caption(image_orig, (x1, y1, x2, y2), caption, font)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imwrite(output_path, image_orig)
        print(f'Saved image to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', type=str, default='labels.txt', help='Path to file listing class names')
    parser.add_argument('--output_path', type=str, default='aicandy_output_ntroyvui/image_out.jpg', help='Image file path to save')
    parser = parser.parse_args()

    detect_image(parser.image_path, parser.model_path, parser.class_list, parser.output_path)
