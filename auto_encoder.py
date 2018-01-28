from keras.layers import Input, Dense
from keras.models import Model 
#Importing the layers and the models to auto_encode it
from keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt

#setting up the encoding dimension
encoding_dim=32

#setting the Input image placeholder 
input_img=Input(shape=(784,))

encoded=Dense(encoding_dim,activation='relu')(input_img)

decoded=Dense(784,activation='sigmoid')(encoded)


#Model maps an input of reducing the dimension from input_img to decoded
autoencoder=Model(input_img,decoded)

#Models maps to encoded representation
encodeder=Model(input_img,encoded)

#create a placeholder for the input (32-Dimensional)
encoded_input=Input(shape=(encoding_dim,))

#retirve the last layer to decode the autoencoder model
decoder_layer=autoencoder.layers[-1]

#create the decoder model
decoder=Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')


#Initailzing the datasets
(x_train,_),(x_test,_)=mnist.load_data()

#segregating the train and test data in correct format
x_train=x_train.astype('float32') / 255
x_test=x_test.astype('float32') / 255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

#train our model
autoencoder.fit(x_train,x_train,epochs=60,batch_size=256,shuffle=True,validation_data=(x_test,x_test))

#Evaluating the train data 
encoded_imgs=encodeder.predict(x_test)
decoded_imgs=decoder.predict(encoded_imgs)

n=10
plt.figure(figsize=(20,4))
for i in range(n):
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

    # display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()


