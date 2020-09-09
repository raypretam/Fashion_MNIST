import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Loading the data from keras
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Classifying the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scaling down the pixel size
train_images = train_images/255.0
test_images = test_images/255.0
print(train_images)

# Neural Network Architechture
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), 
                          keras.layers.Dense(128, activation="relu"), 
                          keras.layers.Dense(10, activation="softmax")])
model.summary()

# Neural Network Configured
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Neural Network Fit data
model.fit(train_images, train_labels, epochs=25, verbose=1)

# Evatuating the dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

print("Tested accuracy: ", test_acc*100.0)

projectId = os.environ['ProjectId']
print(projectId)

# Predict
prediction = model.predict(test_images)

# Looping and showing the predicted models and the actual models
for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual : " + class_names[test_labels[i]])
    plt.title("Prediction : " + class_names[np.argmax(prediction[i])])
    plt.show()
