import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image

# Define the mask_image function as used in training
def mask_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(img, img, mask=thresh)
    return masked

# Class names
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
IMG_SIZE = 224

# Build the model architecture (same as used in training)
def build_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the model by building architecture and loading weights
model = build_model()
model.load_weights("brain_tumor_densenet121.h5")

# Streamlit app
st.title("Brain Tumor Classification")
st.write("Upload a brain MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Classify button
    if st.button("Classify"):
        st.write("Classifying...")

        # Convert PIL image to numpy array
        img = np.array(image)

        # Convert RGB to BGR if necessary
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Apply masking
        img = mask_image(img)

        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img.astype(np.float32) / 255.0

        # Expand dimensions for batch
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        predicted_class = CLASSES[class_index]
        confidence = prediction[0][class_index] * 100

        # Display results
        st.success(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")