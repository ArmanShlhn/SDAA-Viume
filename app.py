# Import library YOLOv5 yang diperlukan
from PIL import Image
import numpy as np
import streamlit as st
import torch
import os  # Import the os module for path manipulation
from torchvision import transforms

import streamlit as st
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

    for det in pred_img.xyxy[0]:
        # Format of det: [x_min, y_min, x_max, y_max, confidence, class]
        x_min, y_min, x_max, y_max, _, class_id = map(int, det)

        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Optionally, you can display class labels
        label = f"Class: {class_id}"
        draw.text((x_min, y_min - 10), label, fill="red")

    return img_with_boxes



def preprocess_image(image):
    # Auto-orient the image
    image = transforms.functional.autocontrast(image)
    
    # Resize the image to 640x640
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).float()
    img_tensor /= 255.0  # Normalization (adjust as needed)
    
    return img_tensor.unsqueeze(0)  # Add batch dimension

# Fungsi untuk melakukan deteksi objek menggunakan YOLOv5
def perform_object_detection(image):
    # Simpan gambar ke file untuk pengolahan oleh YOLOv5
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Load model YOLOv5 yang telah Anda latih
    model_path = "model/best.pt"  # Sesuaikan dengan lokasi Anda

    # Check if the model file exists
    if not os.path.isfile(model_path):
        st.error(f"Model file not found at path: {model_path}")
        return None  # Return early if the model file is not found

    model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=model_path, force_reload=True)
    model.eval()

    # Load image for object detection
    img_tensor = preprocess_image(image)

    # Lakukan inferensi
    with torch.no_grad():
        results = model(img_tensor)

    # Kembalikan hasil deteksi
    return results

# ... (fungsi lainnya tetap sama)



def page_scanner():
    st.title("YOLOv5 Object Detection with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

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
st.sidebar.image(logo_path, width=150)
selected_page = st.sidebar.radio("Menu", list(pages.keys()))

# Memanggil fungsi halaman yang dipilih
pages[selected_page]()