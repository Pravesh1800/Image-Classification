import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101
import cv2
import imghdr


# Prompt user for the number of classes
num_classes = st.number_input("Enter the number of classes", min_value=1, step=1)

# Building the model

def build_model(num_classes):
    base_model = ResNet101  (weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train(model, data):
    status_text = st.empty()  # Create an empty element for dynamic updating
    for epoc in range(epoch):
        model.fit(data, epochs=1)  # Train for 1 epoch at a time
        status_text.write(f"Epoch {epoc + 1} / {epoch} completed")  # Update the status
    #model.save('image_classification.keras')

def main():
    train_dir = "train_data"
    os.makedirs(train_dir, exist_ok=True) 

    for i in range(num_classes):
        st.header(f"Class {i+1}")
        class_name = st.text_input(f"Enter the name of class {i+1}")
        form_key = f"class_{i}_{class_name}_uploader"
        with st.form(key=form_key):
            file_uploader_key = f"class_{i}_{class_name}_uploader"
            uploaded_files = st.file_uploader(f"Choose images for class {class_name}", key=file_uploader_key, accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                if uploaded_files:
                    st.write("Sample Images:")
                    num_images = min(len(uploaded_files), 5)
                    num_columns = min(num_images, 5)
                    columns = st.columns(num_columns)
                    for i, uploaded_file in enumerate(uploaded_files[:num_images]):
                        columns[i].image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                    class_dir = os.path.join(train_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(class_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())

    dir = [i for i in os.listdir(train_dir)]
    data = image_dataset_from_directory(train_dir, batch_size=16, image_size=(224, 224), class_names=dir)

    # One-hot encode the labels
    def preprocess_data(x, y):
        x = x / 255.0  # Normalize pixel values
        y = tf.one_hot(y, depth=num_classes)  # One-hot encode labels
        return x, y

    data = data.map(preprocess_data)

    model = build_model(num_classes)

    
    st.header(" Training the Model")
    global epoch
    epoch = st.slider("Enter the no of epochs you want to run.")


    ################
    
    
    data_dir = "train_data"
    image_ext = ['jpg','jpeg','png']
    
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir,image_class)):
            image_path = os.path.join(data_dir,image_class,image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_ext:
                    print("image not in extension list",image_path)
                    os.remove(image_path)
            except Exception as e:
                print("Issue with image",image_path) 
        
    ################

    if st.button("Train Model"):
        train(model, data)

    # Testing
    st.title("Prediction")
    option = st.radio("Select Input Option", ("Upload Image", "Use Camera"))
    if option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image for prediction", type=['png', 'jpg', 'jpeg'])
    else:
        uploaded_image = st.camera_input("Capture a image to be predicted")

    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        st.write(prediction)
        st.write(f"Predicted class: {predicted_class_index}")
        st.write(dir[predicted_class_index])

if __name__ == "__main__":
    main()
