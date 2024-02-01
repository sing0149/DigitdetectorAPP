import numpy as np  
import tensorflow  as tf

import matplotlib.pyplot as plt

def normalize_image(train_images,test_images): 
     ## checking the type 
     print(test_images.dtype,train_images.dtype)
     ##checking the range is grayscale (0,255)
     print(test_images.max(),test_images.min(),train_images.max(),train_images.min())

     ##converting the insidesof an array as float32 from int
     train_images=train_images.astype(np.float32)
     test_images=test_images.astype(np.float32)

     #nomralize the range
     train_images=train_images/255.0
     test_images=test_images/255.0

     return train_images, test_images






#loading the mnsit dataset

mnist = tf.keras.datasets.mnist
(train_images,train_labels) , (test_images,test_labels)= mnist.load_data()

# #display using matplotlib
# plt.imshow(train_images[0],cmap=plt.cm.binary)
# plt.show()
# print(train_labels[0])
# print(type(train_images))
# print(type(train_labels))

## calll the functoin
normalized_train_images,normalized_test_images=normalize_image(train_images,test_images)

print(normalized_test_images.dtype,type(normalized_test_images))