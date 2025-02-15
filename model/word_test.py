import requests

url = 'http://127.0.0.1:5000/segment'
image_path = 'Handwritten.jpg'

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())