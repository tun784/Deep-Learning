from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import streamlit as st
import tempfile


model_path = 'model/model.h5'
examples = [f"data/image_test/71Banana02034.png", f"data/image_test/Apple 1.png", f"data/image_test/Kiwi A002.png"]

def load_image(image_file):
    img = Image.open(image_file)
    return img


def predict_class_img_with_img(img_arr):
    class_names = ['Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Persimmon', 
                   'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes', 'muskmelon']

    # load model
    model = load_model(model_path)
    np_img = np.array(img_arr)

    # print(np_img.shape)
    image = np.expand_dims(cv2.resize(np.squeeze(np_img), (160, 160)), axis=0)

    predictions = model.predict(image)
    scores = tf.nn.sigmoid(predictions)
    pred_labels = np.argmax(scores, axis=-1)
    return class_names[int(pred_labels)]


if __name__ == '__main__':
    st.title("Chương trình phân loại trái cây: ")
    f = st.file_uploader("Tải file lên")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ex = load_image(examples[0])
        st.image(ex, width=200)
        if st.button('Ví dụ 1'):
            f = examples[0]

    with col2:
        ex1 = load_image(examples[1])
        st.image(ex1, width=200)
        if st.button('Ví dụ 2'):
            f = examples[1]

    with col3:
        ex2 = load_image(examples[2])
        st.image(ex2, width=200)
        if st.button('Ví dụ 3'):
            f = examples[2]

    if f is not None: 
        img = load_image(f)
        img = np.asarray(img)
        st.image(img, width=400)
        st.text("Đang dự đoán ....")
        class_name = predict_class_img_with_img(img)
        st.write("Kết quả phân loại trái cây: ")
        st.header(class_name)