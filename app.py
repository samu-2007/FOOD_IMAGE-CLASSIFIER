import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image

# Page settings
st.set_page_config(page_title="Food Nutrition Detector", page_icon="🍔")

st.title("🍔 Food Nutrition Detection using YOLOv8")
st.write("Upload a food image to predict the food and view its nutrition information.")

# Load model
model = YOLO("best.pt")

# Load nutrition table
nutrition_df = pd.read_excel("Nutrition_Table_15_Food_Classes.xlsx")

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):

        results = model.predict(image)

        predicted_class = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf)

        st.success(f"Predicted Food: {predicted_class.title()}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        food_name = predicted_class.replace("_", " ").title()

        nutrition = nutrition_df[
            nutrition_df["Food Name"].str.lower() == food_name.lower()
        ]

        if not nutrition.empty:

            st.subheader("🥗 Nutrition Information")

            st.dataframe(nutrition, use_container_width=True)

        else:

            st.warning("Nutrition information not found.")
            