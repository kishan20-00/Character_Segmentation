import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
WORD_FOLDER = './segmented_words'
CHAR_FOLDER = './segmented_chars'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORD_FOLDER, exist_ok=True)
os.makedirs(CHAR_FOLDER, exist_ok=True)

def segment_words(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh, kernel_line, iterations=1)
    contours_line, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_lines = sorted(contours_line, key=lambda ctr: cv2.boundingRect(ctr)[1])

    kernel_word = np.ones((3, 15), np.uint8)
    dilated_word = cv2.dilate(thresh, kernel_word, iterations=1)

    word_paths = []
    word_index = 1

    for line in sorted_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_word[y:y + h, x:x + w]

        contours_word, _ = cv2.findContours(roi_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_words = sorted(contours_word, key=lambda c: cv2.boundingRect(c)[0])

        for word in sorted_words:
            if cv2.contourArea(word) < 400:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            word_img = img[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
            word_path = os.path.join(WORD_FOLDER, f'word_{word_index}.png')
            cv2.imwrite(word_path, cv2.cvtColor(word_img, cv2.COLOR_RGB2BGR))
            word_paths.append(word_path)
            word_index += 1

    return word_paths

def segment_characters(image_path, word_index):
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

    bg_color = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    if np.sum(bg_color == 255) > np.sum(bg_color == 0):
        final_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        final_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    morp = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, (5, 5))
    vertical_hist = np.sum(morp, axis=0) / 255

    char_start, char_end = [], []
    in_char = False

    for i, val in enumerate(vertical_hist):
        if val > 3 and not in_char:
            char_start.append(i)
            in_char = True
        elif val < 3 and in_char:
            char_end.append(i)
            in_char = False

    if len(char_start) > len(char_end):
        char_start = char_start[:len(char_end)]

    char_paths = []
    for idx, (start, end) in enumerate(zip(char_start, char_end)):
        char_img = morp[:, start:end]
        char_img = cv2.bitwise_not(char_img)
        char_path = os.path.join(CHAR_FOLDER, f'word_{word_index}_char_{idx}.png')
        cv2.imwrite(char_path, char_img)
        char_paths.append(char_path)

    return char_paths

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    word_paths = segment_words(image_path)

    all_chars = {}
    for idx, word_path in enumerate(word_paths, 1):
        chars = segment_characters(word_path, idx)
        all_chars[f'word_{idx}'] = chars

    return jsonify({'words_segmented': len(word_paths), 'character_paths': all_chars})

if __name__ == '__main__':
    app.run(debug=True)
