import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np


#Streamlit app title
st.title("TEXT EXTRACTION FROM IMAGES")

#File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Display uploaded image
    st.image(uploaded_file, width=300)

    #Extract text from image
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)

    #Display extracted text
    st.header("Extracted Text")
    st.write(text)

    #preprocessing work
    def preprocess_image(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        #apply binary thresholding
        _, thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh
    
    #USE PRE-PROCESSED IMAGE FOR BETTER OCR ACCURACY
    preprocessed_image = preprocess_image(image)
    text_preprocessed = pytesseract.image_to_string(preprocessed_image)

    #DISPLAY EXTRACTED TEXT USING PRE-PROCESSED IMAGE
    st.header("Extracted Text (Pre-processed Image)")
    st.write(text_preprocessed)
    