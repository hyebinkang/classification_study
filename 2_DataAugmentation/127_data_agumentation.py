#Single Image augmentation
from keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Construct an instance of the ImageDataGenerator class, ImageDataGenerator 클래스 인스턴스 생성,
# Pass the augmentation parameters through the constructor. 기능 보강 매개변수 전달

datagen = ImageDataGenerator(
    rotation_range=45,  # Random rotation between 0 and 45(limit)
    width_shift_range=0.2,  # % shift
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant', cval=125)  # Also try nearest, constant, reflect, wrap # cval = 125: gray

######################################################################
# Loading a single image for demonstration purposes.
# Using flow method to augment the image

# Loading a sample image
# Can use any library to read images but they need to be in an array form
# If using keras load_img convert it to an array first

x = io.imread('./choonbae.jpg')  # Array with shape (256, 256, 3)

# Reshape the input image because ...
# x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple. -> datagenerator 는 4차원 numpy 배열, 튜플로 들어와야 한다.
# First element represents the number of images
x = x.reshape((1,) + x.shape)  # Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(x, batch_size=16,                         #.flow -> single images
                          save_to_dir='augmented',
                          save_prefix='aug',
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

####################################################################
# Multiple images.
# Manually read each image and create an array to be supplied to datagen via flow method
dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'test_folder/'
SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)

# sub directory 있을 때 접근 방법 -> flow_from_directory 로
# Let us save images to get a feel for the augmented images.
# Create an iterator either by using image dataset in memory (using flow() function)
# or by using image dataset from a directory (using flow_from_directory)
# from directory can be useful if subdirectories are organized by class

# Generating and saving 10 augmented samples
# using the above defined parameters.
# Again, flow generates batches of randomly augmented images
"""   
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='augmented', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  
"""

#####################################################################
# Multiclass. Read dirctly from the folder structure using flow_from_directory

i = 0
for batch in datagen.flow_from_directory(directory='monalisa_einstein/',            #multi image 는 flow_from_direcotry
                                         batch_size=16,
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir='augmented',
                                         save_prefix='aug',
                                         save_format='png'):
    i += 1
    if i > 31:
        break

    # Creates 32 images for each class.

# Once data is augmented, you can use it to fit a model via: fit.generator
# instead of fit()
# model =
# fit model on augmented data
# model.fit_generator(datagen.flow(x))