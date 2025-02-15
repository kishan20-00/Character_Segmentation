import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
os.makedirs('segmented_words', exist_ok=True)

def segment_words(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Line segmentation
    kernel_line = np.ones((3, 85), np.uint8)
    dilated_line = cv2.dilate(thresh, kernel_line, iterations=1)
    contours_line, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours_line, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Word segmentation
    kernel_word = np.ones((3, 15), np.uint8)
    dilated_word = cv2.dilate(thresh, kernel_word, iterations=1)

    words_saved = []
    word_index = 1

    for line in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_word[y:y + h, x:x + w]

        contours_word, _ = cv2.findContours(roi_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_words = sorted(contours_word, key=lambda c: cv2.boundingRect(c)[0])

        for word in sorted_contours_words:
            if cv2.contourArea(word) < 400:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            word_img = img[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
            word_filename = f'segmented_words/word_{word_index}.png'
            cv2.imwrite(word_filename, cv2.cvtColor(word_img, cv2.COLOR_RGB2BGR))
            words_saved.append(word_filename)
            word_index += 1

    return words_saved

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = 'uploaded_image.png'
    image.save(image_path)

    words = segment_words(image_path)

    if not words:
        return jsonify({'message': 'No words found'}), 404

    return jsonify({'message': f'{len(words)} words segmented', 'words': words})

if __name__ == '__main__':
    app.run(debug=True)
