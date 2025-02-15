import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def segment_characters(image_path):
    img = cv2.imread(image_path)
    h, w, c = img.shape
    ar = w/h
    targetWidth = 600
    targetHeight = int(targetWidth / ar)
    img = cv2.resize(img, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)

    img_blf = cv2.bilateralFilter(img, 5, 7, 7)
    lab = cv2.cvtColor(img_blf, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    cimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    def check_bg_color(image):
        check = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
        u, c = np.unique(check, return_counts=True)
        srt = sorted(dict(zip(u, c)).items(), key=lambda x: x[1], reverse=True)
        return srt[0][0]

    bg_color = check_bg_color(gray)

    if bg_color == 255:
        final_img2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        final_img2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    morp = cv2.morphologyEx(final_img2, cv2.MORPH_OPEN, (5, 5))

    horizontal_hist = np.sum(morp, axis=1) / 255
    half = len(horizontal_hist) // 2
    threshold = 40

    left_limit = next((i for i in range(half - 1, -1, -1) if horizontal_hist[i] < threshold), 0) + 1
    right_limit = next((i for i in range(half, len(horizontal_hist)) if horizontal_hist[i] < threshold), len(horizontal_hist) - 1) - 1

    morp2 = morp[left_limit:right_limit + 1]

    vertical_hist = np.sum(morp2, axis=0) / 255
    check_left, check_right = [], []

    def find_segments(start, arr):
        for i in range(start, len(arr)):
            if arr[i] > 3:
                for j in range(i + 1, len(arr)):
                    if arr[j] < 3 and j - i > 3:
                        check_left.append(i)
                        check_right.append(j)
                        find_segments(j + 1, arr)
                        return
    find_segments(0, vertical_hist)

    segmented_images = []
    for idx, (l, r) in enumerate(zip(check_left, check_right)):
        char_img = morp2[:, l:r]
        char_path = os.path.join(OUTPUT_FOLDER, f'char_{idx}.png')
        cv2.imwrite(char_path, char_img)
        segmented_images.append(char_path)

    return segmented_images

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        segmented_paths = segment_characters(image_path)
        return jsonify({'segmented_images': segmented_paths})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
