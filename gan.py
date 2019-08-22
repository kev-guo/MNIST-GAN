import numpy as np
import keras as ker
from tqdm import tqdm

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, LeakyReLU, Conv2DTranspose, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras import initializers
from numpy import expand_dims, ones, zeros, vstack
from numpy.random import rand, randint, randn
import matplotlib.pyplot as plt

#import MNIST database
from keras.datasets import mnist

#use theano for tensor shape, try "image_dim_ordering" for updated API
K.set_image_dim_ordering('th')

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

#for reproducible results
seed = 1
np.random.seed(seed)
latent_dim = 100

discriminator = Sequential()
discriminator.add(Dense(1024,input_dim=784))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))  
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))   
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2)) 
discriminator.add(Dense(units=1, activation='sigmoid')) 
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    
generator = Sequential()
generator.add(Dense(256,input_dim=latent_dim))
generator.add(LeakyReLU(0.2))    
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))   
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

discriminator.trainable = False
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

discrim_losses = []
gen_losses = []

#print generated images
def printGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    #format array of pictures
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

#plot each batch's discriminator and generator losses
def plotLoss(epoch):
    plt.figure(figsize=(9, 6))
    plt.legend()
    plt.plot(discrim_losses, label='Discriminator loss')
    plt.plot(gen_losses, label='Generator loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)
    
#train gan
def train(epochs=1, batch_size=128):
    batch_count = X_train.shape[0] / batch_size
    print('Epochs:', epochs)
    print('Batch size:', batch_size)
    print('Batches per epoch:', batch_count)

    for e in range(1, epochs+1):
        print('Epoch %d' % e)
        
        for _ in tqdm(range(int(batch_count))):
            #create randomized noise and images
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

            #generate fake images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            #label fake or not
            Y = np.zeros(2*batch_size)
            Y[:batch_size] = 0.9

            #train discriminator and generator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, Y)

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        #store epoch loss
        discrim_losses.append(d_loss)
        gen_losses.append(g_loss)
        
        #print images and save weights per array
        if e == 1 or e % 25 == 0:
            printGeneratedImages(e)
            generator.save('model_parameters/gan_generator_epoch_%d.h5' % e)
            discriminator.save('model_parameters/gan_discriminator_epoch_%d.h5' % e)

    #plot losses
    plotLoss(e)

#run
if __name__ == '__main__':
    train(200, 128)
