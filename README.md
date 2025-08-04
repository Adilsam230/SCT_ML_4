# SCT_ML_4
This project uses a Convolutional Neural Network (CNN) to build and train a simple image classifier to recognize different hand signs. The script automates the entire machine learning workflow from data loading to model saving.

The script automates the following steps:
1.) Data Loading & Preprocessing: It preprocesses all images by resizing them to a uniform 128x128 pixels and rescaling the pixel values to a 0-1 range for the model.
2.) The data is automatically split into a training set (80%) and a validation set (20%), allowing the model to be evaluated on unseen data during training.
3.) Model Architecture: It defines a simple sequential CNN model. The architecture consists of two convolutional blocks (each with a Conv2D and MaxPooling2D layer) to extract basic features
4.) Training: The model is compiled with the standard Adam optimizer and categorical_crossentropy loss function. It is then trained on the dataset for a fixed number of epochs.
5.) Saving the Model: After training is complete, the script saves the final trained model to a file named HandGestureModel

You will need Python 3 and the following libraries:
1.) TensorFlow
2.) Pillow (a dependency for Keras image processing)
