import requests

url = 'http://localhost:5000/segment'
image_path = 'segmented_words/word_1.png'

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())
