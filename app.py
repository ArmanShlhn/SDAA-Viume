# Import library YOLOv5 yang diperlukan
from PIL import Image
import numpy as np
import streamlit as st
import torch
import os  # Import the os module for path manipulation
from yolov5.models.yolo import Model


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

    model = torch.load(model_path, map_location='cpu')['model'].float()  # Pemrosesan di CPU, model diubah ke float
    model.eval()

    # Load image for object detection
    img_tensor = model.preprocess(image)
    img_tensor = img_tensor.float()  # YOLOv5 memerlukan tipe data float
    img_tensor /= 255.0  # Normalisasi

    # Lakukan inferensi
    with torch.no_grad():
        results = model(img_tensor)

    # Kembalikan hasil deteksi
    return results

# Fungsi halaman Scanner yang diperbarui
def page_scanner():
    st.title("Scanner")
    
    # Widget untuk mengunggah gambar
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Menggunakan PIL untuk membaca gambar
        image = Image.open(uploaded_image)

        # Menampilkan gambar
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Lakukan deteksi objek menggunakan YOLOv5
        results = perform_object_detection(image)

        # Check if the results are available
        if results is not None:
            # Tampilkan hasil deteksi
            for result in results.names[0]:
                st.write(f"Detected: {result} with confidence {result.confidence.item():.2f}")

            # Pemrosesan gambar (contoh: konversi ke array NumPy)
            image_array = np.array(image)
            st.write("Image Shape:", image_array.shape)
        
    st.write("Berikut adalah hasil dari skrinning sel kanker rahim")


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