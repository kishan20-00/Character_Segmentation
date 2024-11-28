import streamlit as st
import cv2
import numpy as np
from pre_processing import preProcessing

# Function to read an uploaded file and convert it to a format OpenCV can process
def read_uploaded_file(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

# Streamlit app layout
st.title("Character Segmentation App")
st.markdown("Upload a handwritten image, and this app will process it to segment the characters.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Convert uploaded file to OpenCV format
    image = read_uploaded_file(uploaded_file)
    
    # Process the image using preProcessing
    st.write("Processing the image...")
    processed_image = preProcessing(image)
    
    # Display the processed image
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
    
    # Extract segmented characters
    st.write("Displaying segmented characters...")
    
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        character = image[y:y+h, x:x+w]
        st.image(cv2.cvtColor(character, cv2.COLOR_BGR2RGB), caption=f"Character {i+1}", use_container_width=False)
