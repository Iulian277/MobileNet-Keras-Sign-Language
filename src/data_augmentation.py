import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(rotation_range = 10,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             shear_range = 0.15,
                             zoom_range = 0.1,
                             channel_shift_range = 10.,
                             horizontal_flip = True)


# Save the augmented data
# 10 dirs
for i in range(10):

    # Delete the old aug data
    for file in glob.glob(f'../data/Sign-Language-Digits-Dataset/train/{i}/aug*'):
        os.remove(file)

    # Pick 15 images from each dir
    num_of_samples = 15
    samples_to_aug = random.sample(os.listdir(f'../data/Sign-Language-Digits-Dataset/train/{i}'), num_of_samples)

    for sample in samples_to_aug:

        image_path = f'../data/Sign-Language-Digits-Dataset/train/{i}/' + sample
        image = np.expand_dims(plt.imread(image_path), 0)
        datagen.fit(image)

        # Save 10 augmented images for each sample
        save_path = f'../data/Sign-Language-Digits-Dataset/train/{i}'
        for x, val in zip(datagen.flow(image, save_to_dir = save_path, save_prefix = 'aug-image-', save_format = 'PNG'), range(10)):
            pass







