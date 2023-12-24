import streamlit as st
import numpy as np
import os  # Import the os module for path manipulation
from torchvision import transforms
from PIL import Image, ImageDraw
import torch
import io
import json

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Kuliah/model/SDAA/model/best.pt')

def predict_image(file):
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))

    # Perform prediction
    pred_img = model(img)

    # Render predicted image with bounding boxes
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Dictionary untuk memetakan ID kelas ke nama kelas
    class_mapping = {
        0: "ASC-H",
        1: "ASCH-US",
        2: "HSIL",
        3: "LSIL",
        4: "Normal",
        5: "SCC",
    # Tambahkan mapping lain sesuai kebutuhan
    }

# Melakukan iterasi pada objek pred_img.xyxy[0]
    detections_count = len(pred_img.xyxy[0])
    class_count = len(class_mapping)

    # Inisialisasi dictionary untuk menyimpan jumlah deteksi setiap kelas
    class_detection_count = {class_id: 0 for class_id in class_mapping}

    for det in pred_img.xyxy[0]:
        # Format det: [x_min, y_min, x_max, y_max, confidence, class]
        x_min, y_min, x_max, y_max, _, class_id = map(int, det)

        # Increment count untuk kelas ini
        class_detection_count[class_id] += 1

        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Mendapatkan nama kelas berdasarkan ID kelas
        class_name = class_mapping.get(class_id, f"Unknown Class {class_id}")

        # Display class labels
        label = f"Class: {class_name}"
        draw.text((x_min, y_min - 10), label, fill="red")

    # Menghitung persentase deteksi untuk setiap kelas
    class_percentage = {class_id: (count / detections_count) * 100 for class_id, count in class_detection_count.items()}

    # Menampilkan persentase deteksi untuk setiap kelas
    for class_id, percentage in class_percentage.items():
        class_name = class_mapping.get(class_id, f"Unknown Class {class_id}")
        st.write(f"{class_name}: {percentage:.2f}%")
        
    return img_with_boxes


def page_scanner():
    st.title("YOLOv5 Object Detection with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Cervical cancer cell screening results by VIUME")

        # Perform prediction
        pred_image = predict_image(uploaded_file)

        # Display the predicted image
        st.image(pred_image, caption="Predicted Image.", use_column_width=True)

if __name__ == "__page_scanner__":
    page_scanner()



# Fungsi untuk membuat halaman
def page_home():
    st.title("WELCOME TO VIUME")
    st.write("DIGITAL PATHOLOGY PLATFORM FOR CERVICAL CANCER DETECTION")
    st.write("Aplikasi yang dirancang untuk memanfaatkan teknologi patologi digital guna meningkatkan deteksi dan diagnosis kanker serviks")

def page_about():
    st.title("About")
    st.write("Ini adalah halaman tentang kami")

def page_contact():
    st.title("Contact")
    st.write("Hubungi kami di sini")

# Dictionary untuk menyimpan halaman
pages = {
    "Home": page_home,
    "Scanner": page_scanner,
    "About": page_about,
    "Contact": page_contact,
}

# Layout navbar
logo_path = "img\logoviume-removebg-preview.png"
st.sidebar.image(logo_path, width=250)
selected_page = st.sidebar.selectbox("", list(pages.keys()))

# Memanggil fungsi halaman yang dipilih
pages[selected_page]()