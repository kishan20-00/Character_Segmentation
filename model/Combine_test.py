import requests

url = 'http://127.0.0.1:5000/segment'
image_path = 'Handwritten.jpg'  # Replace with your image path

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.json()}")
