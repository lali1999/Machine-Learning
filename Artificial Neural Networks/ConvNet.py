# Building Convolutional Neural Networks to Classify the Dog and Cat Images. This is a Binary Classification Model i.e. 0 or 1
# Used Dataset -- a Subset (10,000) Images ==> (8,000 for training_set: 4,000 Dogs and 4,000 Cats)
# (2,000 for test_set: 1,000 Dogs and 1,000 Cats of Original Dataset (25,000 images) of Dogs vs. Cats | Kaggle
# Original Dataset link ==> https://www.kaggle.com/c/dogs-vs-cats/data
# You might use 25 or more epochs and 8000 Samples per epoch

# Installing Theano
# Installing Tensorflow
# Installing Keras

# Part 1 - Building the ConvNet

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the ConvNet
classifier = Sequential()

# Step 1 - Building the Convolution Layer
classifier.add(Convolution2D(32, 3,input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Building the Pooling Layer (max pooling)
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding The Second Convolutional Layer
classifier.add(Convolution2D(32, 3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Building the Flattening Layer
classifier.add(Flatten())
 
# Step 4 - Building the Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ConvNet
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the ConvNet to the Images
from keras.preprocessing.image import ImageDataGenerator
# ..... Fill the Rest (a Few Lines of Code!)

training_data_generator= ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_data_set = training_data_generator.flow_from_directory(
       'D:/Kennesaw State/Summer 2022 semester/Machine Learning/Programming Assignments/Programming Assignment-03-CS7267-Ayush Patel/ConvNet_dataset/training_set',
        target_size=(64,64),
        batch_size=35,
        class_mode='binary')

test_data_generator = ImageDataGenerator(rescale=1./255)    
testing_data_set = test_data_generator.flow_from_directory(
         'D:/Kennesaw State/Summer 2022 semester/Machine Learning/Programming Assignments/Programming Assignment-03-CS7267-Ayush Patel/ConvNet_dataset/test_set',
        target_size=(64,64),
        batch_size=35,
        shuffle=False, 
        class_mode='binary')

#fitting our dataset of images.
classifier.fit(x=training_data_set,validation_data= testing_data_set,epochs=25) 

#predicting whether it is a Cat or a Dog
import numpy as np
from tensorflow.keras.preprocessing import image

test_case_1 = image.load_img('test_image_3.jpg', target_size=(64,64))
test_case_1 = image.img_to_array(test_case_1) 
test_case_1 = np.expand_dims(test_case_1, axis = 0)
pred_ans = classifier.predict(test_case_1)
training_data_set.class_indices

if pred_ans[0][0] == 1:
    predict = 'Dog'
else:
    predict = 'Cat'
    
print("The given image is of a: "+predict)
