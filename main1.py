import streamlit as st
from PIL import Image, ImageDraw
import torch
import io
import json

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Fathur Files/Semester 7/SDAA/model/best.pt')

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


def main():
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

if __name__ == "__main__":
    main()
