
# Pneumonia Detection using Deep Learning

This project uses deep learning techniques to detect pneumonia from chest X-ray images. A pre-trained MobileNetV2 model is used for transfer learning, allowing us to classify X-ray images into two categories: **Normal** and **Pneumonia**.

## Table of Contents
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Preprocessing and Augmentation](#preprocessing-and-augmentation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Prediction](#prediction)
- [Conclusion](#conclusion)

## Dataset
The dataset used for this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. The dataset consists of images from two categories:
- **Normal**: Healthy X-rays
- **Pneumonia**: X-rays showing symptoms of pneumonia

### Directory Structure
- `train/`: Contains training images
  - `NORMAL/`: Images of normal X-rays
  - `PNEUMONIA/`: Images of pneumonia X-rays
- `val/`: Validation images
- `test/`: Testing images

## Environment Setup
To set up the environment:
1. Install the required libraries:
    ```bash
    pip install tensorflow keras matplotlib numpy scikit-learn
    ```
2. Download the dataset from Kaggle:
    ```bash
    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
    unzip chest-xray-pneumonia.zip
    ```

## Preprocessing and Augmentation
Data augmentation was used to artificially increase the dataset size and improve generalization. `ImageDataGenerator` was used to apply transformations like rotation, shear, zoom, and horizontal flip.

Example code to create the training data generator:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)
```

## Model Architecture
A **MobileNetV2** model is used with **transfer learning** for feature extraction. The base model is pretrained on the ImageNet dataset and the classifier head is modified for binary classification.

Example code to set up the model:
```python
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Training and Evaluation
The model was compiled using the `Adam` optimizer and `binary_crossentropy` loss function.

Example code for training:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[callback])
```

After training, the model was evaluated on a test set of 624 images. The accuracy achieved was around **91%**.

## Prediction
To predict an image, the model takes an X-ray image, resizes it, and classifies it into one of the two categories: **Normal** or **Pneumonia**.

Example function for predicting an image:
```python
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    print(f"Prediction: {label} ({prediction * 100:.2f}% confidence)")
```
##### Built ML models (RandomForest, GradientBoosting) achieving R² 0.82 and cut prediction error by 18% .
##### Analyzed 10,000+ transportation records; key drivers- fuel type (35% impact), distance (27%), and vehicle (22%).
##### Delivered insights on high emission vehicles 2.3× more carbon than their alternatives, guiding reduction strategies.

## Conclusion
This project demonstrates the use of transfer learning for pneumonia detection. By leveraging a pre-trained model (MobileNetV2), the system is able to accurately classify chest X-ray images as either normal or pneumonia with high confidence.

## Future Improvements
- Fine-tune the base model to improve performance.
- Integrate the model into a web or mobile application.


