import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = r"C:\dataset"
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(train_generator.num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Starting training on a simple CNN... 🚀")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
model.save('Hand_Gesture_Model.h5')
print("\nModel saved successfully. ✨")