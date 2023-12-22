import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
import numpy as np


# Fungsi untuk membuat halaman
def page_home():
    st.title("WELCOME TO VIUME")
    st.write("DIGITAL PATHOLOGY PLATFORM FOR CERVICAL CANCER DETECTION")
    st.write("Aplikasi yang dirancang untuk memanfaatkan teknologi patologi digital guna meningkatkan deteksi dan diagnosis kanker serviks")

def page_scanner():
    st.title("Scanner")
        # Widget untuk mengunggah gambar
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Menggunakan PIL untuk membaca gambar
        image = Image.open(uploaded_image)

        # Menampilkan gambar
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Pemrosesan gambar (contoh: konversi ke array NumPy)
        image_array = np.array(image)
        st.write("Image Shape:", image_array.shape)
    st.write("Berikut adalah hasil dari skrinning sel kanker rahim")

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