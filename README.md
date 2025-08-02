# SCT_ML_4
This project uses a Convolutional Neural Network (CNN) to build and train a hand gesture recognition model. The script automatically loads images from a dataset, augments the training data to improve robustness, and trains a deep learning model to classify different hand gestures. The final trained model is then saved for future use.

The script automates the entire deep learning workflow:

1. Data Loading & Augmentation: It loads images from a dataset directory using ImageDataGenerator. To make the model more robust and prevent overfitting, it applies real-time data augmentation (rotation, shifting, zooming, etc.) to the training images.
2. Dataset Splitting: The data is automatically split into a training set (80%) and a validation set (20%).
3. Model Architecture: It defines a sequential CNN model with multiple convolutional and pooling layers to extract features from the images, followed by dense layers for classification. A dropout layer is included to reduce overfitting.
4. Training: The model is compiled with the Adam optimizer and trained on the dataset. It uses EarlyStopping to monitor the validation loss and stop training if the model's performance on the validation set does not improve, saving the best version of the model.
5. Saving the Model: After training, the script saves the final trained model to a file named hand_gesture_model.h5 and also saves the class names to class_names.txt so they can be easily loaded for prediction later.

You will need Python 3 and the following libraries:
1. Tensorflow
2. Pillow (a dependency for image processing in Keras)
