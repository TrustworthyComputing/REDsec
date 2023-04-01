#!/usr/bin/env python
# coding: utf-8

#!pip install larq
#!pip install tf-nightly
#!pip install --upgrade tensorflow
#!pip install --upgrade tensorflow-gpu
#!pip install keras==2.4.0 #2.3.1
#!pip install tensorflow-cpu==2.4.0

import tensorflow as tf
from tensorflow import keras 
import larq as lq
from keras_preprocessing.image import ImageDataGenerator


#TODO: Download and prepare the datasets
    #MNIST is loaded here as an example. 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

    #Preprocessing: Any added preprocessing must also be done in main.cpp
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

val_images, val_labels = test_images[:-1000], test_labels[:-1000]
test_images, test_labels = test_images[-1000:], test_labels[-1000:]
#END TODO: Download and prepare the datasets

# Here are the default options
in_args = dict(input_quantizer= lq.quantizers.NoOp(precision=8))
reluArgs = dict(input_quantizer= lq.quantizers.DoReFa(k_bit=8))
signArgs = dict(input_quantizer= "ste_sign")

model = tf.keras.models.Sequential()

##START NETWORK ARCHITECTURE
model.add(tf.keras.Input((28,28,1)))
model.add(tf.keras.layers.AveragePooling2D((2,2), strides=(2,2), padding="valid"))
model.add(tf.keras.layers.Flatten())
model.add(lq.layers.QuantDense(1024 ,kernel_quantizer=lq.quantizers.SteTern(threshold_value=0.1),kernel_constraint="weight_clip", use_bias=False,input_quantizer= lq.quantizers.NoOp(precision=4) ,))
model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon= 0.001, scale=False))
model.add(lq.layers.QuantDense(1024 ,kernel_quantizer=lq.quantizers.SteTern(threshold_value=0.1),kernel_constraint="weight_clip", use_bias=False,input_quantizer= lq.quantizers.DoReFa(k_bit=4) ,))
model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon= 0.001, scale=False))
model.add(lq.layers.QuantDense(1024 ,kernel_quantizer=lq.quantizers.SteTern(threshold_value=0.1),kernel_constraint="weight_clip", use_bias=False,input_quantizer= lq.quantizers.DoReFa(k_bit=4) ,))
model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon= 0.001, scale=False))
model.add(lq.layers.QuantDense(10 ,kernel_quantizer=lq.quantizers.SteTern(threshold_value=0.1),kernel_constraint="weight_clip", use_bias=False,input_quantizer= lq.quantizers.DoReFa(k_bit=4) ,))
model.add(tf.keras.layers.Activation("softmax"))
#END NETWORK ARCHITECTURE

lq.models.summary(model)

#Basic Training. You can edit the training for higher accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

max_acc = 0
val_acc = 0
for i in range(200):
    print(i)
    model.fit(train_images, train_labels, batch_size=64, epochs=1, verbose=0, shuffle=True)
    val_loss, val_acc = model.evaluate(val_images, val_labels)
    if val_acc > max_acc:
        max_acc = val_acc
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print("New Max!!!:", max_acc, test_acc)
        model.save('best')
model.save('final')
