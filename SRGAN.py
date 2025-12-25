"""
srgan.py

Implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)
in Keras.

Original Paper: https://arxiv.org/abs/1609.04802

Tutorial Videos (by DigitalSreeni):
- Part 1: https://youtu.be/nbRkLE2fiVI
- Part 2: https://youtu.be/1HqjPqNglPc

This code demonstrates training an SRGAN model on low-resolution (32x32) and high-resolution (128x128)
versions of images (originally from MIR Flickr dataset or similar).

Note: This is an educational implementation. For production or research use, consider modern
PyTorch-based implementations (e.g., ESRGAN).

Author: Adapted from DigitalSreeni's tutorial
License: MIT (feel free to use and modify)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import (Conv2D, PReLU, BatchNormalization, Flatten,
                          UpSampling2D, LeakyReLU, Dense, Input, Add)
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# ----------------------------------------------------------------------
# Define building blocks for the Generator
# ----------------------------------------------------------------------
def res_block(ip):
    """Residual block used in the Generator"""
    res_model = Conv2D(64, (3, 3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)

    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)

    return Add()([ip, res_model])


def upscale_block(ip):
    """Upsampling block (x2) using Conv2D + UpSampling2D"""
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    return up_model


def create_generator(gen_input, num_res_blocks=16):
    """Build the SRGAN Generator model"""
    layers = Conv2D(64, (9, 9), padding="same")(gen_input)
    layers = PReLU(shared_axes=[1, 2])(layers)
    temp = layers

    for _ in range(num_res_blocks):
        layers = res_block(layers)

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = Add()([layers, temp])

    layers = upscale_block(layers)  # x2
    layers = upscale_block(layers)  # x4 total

    output = Conv2D(3, (9, 9), padding="same", activation='tanh')(layers)

    return Model(inputs=gen_input, outputs=output)


# ----------------------------------------------------------------------
# Define building blocks for the Discriminator
# ----------------------------------------------------------------------
def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)

    disc_model = LeakyReLU(alpha=0.2)(disc_model)

    return disc_model


def create_discriminator(disc_input):
    """Build the SRGAN Discriminator model"""
    df = 64

    d1 = discriminator_block(disc_input, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df * 2)
    d4 = discriminator_block(d3, df * 2, strides=2)
    d5 = discriminator_block(d4, df * 4)
    d6 = discriminator_block(d5, df * 4, strides=2)
    d7 = discriminator_block(d6, df * 8)
    d8 = discriminator_block(d7, df * 8, strides=2)

    flattened = Flatten()(d8)
    d9 = Dense(df * 16)(flattened)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(inputs=disc_input, outputs=validity)


# ----------------------------------------------------------------------
# VGG19 for perceptual loss (features from layer before 5th maxpool)
# ----------------------------------------------------------------------
def build_vgg(hr_shape):
    """Extract features from VGG19 (layer 10 output = after activation before 5th maxpool)"""
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    vgg.trainable = False
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


# ----------------------------------------------------------------------
# Combined GAN model (Generator + frozen Discriminator + VGG)
# ----------------------------------------------------------------------
def create_gan(generator, discriminator, vgg, lr_input, hr_input):
    generated_img = generator(lr_input)
    generated_features = vgg(generated_img)

    discriminator.trainable = False
    validity = discriminator(generated_img)

    return Model(inputs=[lr_input, hr_input], outputs=[validity, generated_features])


# ----------------------------------------------------------------------
# Main script (data loading, training, testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------- Data Loading ---------------------
    data_dir = "data"  # Adjust to your folder structure
    lr_dir = os.path.join(data_dir, "lr_images")
    hr_dir = os.path.join(data_dir, "hr_images")

    n_images = 5000  # Number of images to load (for demo)

    lr_list = sorted(os.listdir(lr_dir))[:n_images]
    hr_list = sorted(os.listdir(hr_dir))[:n_images]

    lr_images = [cv2.cvtColor(cv2.imread(os.path.join(lr_dir, img)), cv2.COLOR_BGR2RGB) for img in lr_list]
    hr_images = [cv2.cvtColor(cv2.imread(os.path.join(hr_dir, img)), cv2.COLOR_BGR2RGB) for img in hr_list]

    lr_images = np.array(lr_images) / 255.0
    hr_images = np.array(hr_images) / 255.0

    # Train-test split
    lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)

    # Shapes
    lr_shape = lr_train[0].shape
    hr_shape = hr_train[0].shape

    # Inputs
    lr_input = Input(shape=lr_shape)
    hr_input = Input(shape=hr_shape)

    # Models
    generator = create_generator(lr_input, num_res_blocks=16)
    discriminator = create_discriminator(hr_input)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    vgg = build_vgg(hr_shape)
    vgg.trainable = False

    gan = create_gan(generator, discriminator, vgg, lr_input, hr_input)
    gan.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")

    # --------------------- Training ---------------------
    batch_size = 1
    epochs = 5  # Increase significantly for real training (e.g., 100+)

    # Pre-create batches for speed
    train_lr_batches = [lr_train[i:i + batch_size] for i in range(0, len(lr_train), batch_size)]
    train_hr_batches = [hr_train[i:i + batch_size] for i in range(0, len(hr_train), batch_size)]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        g_losses = []
        d_losses = []

        for b in tqdm(range(len(train_hr_batches))):
            lr_batch = train_lr_batches[b]
            hr_batch = train_hr_batches[b]

            fake_imgs = generator.predict_on_batch(lr_batch)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train Discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(hr_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator (via GAN)
            discriminator.trainable = False
            image_features = vgg.predict(hr_batch)
            g_loss, _, _ = gan.train_on_batch([lr_batch, hr_batch], [real_labels, image_features])

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        print(f"  Generator Loss: {np.mean(g_losses):.4f} | Discriminator Loss: {np.mean(d_losses):.4f}")

        if (epoch + 1) % 10 == 0:
            generator.save(f"models/gen_epoch_{epoch + 1}.h5")

    # --------------------- Testing ---------------------
    # Load a saved generator (replace with your trained model)
    # generator = load_model("models/gen_epoch_10.h5", compile=False)

    # Example on test set
    ix = random.randint(0, len(lr_test) - 1)
    src = lr_test[ix:ix+1]
    gen = generator.predict(src)
    tar = hr_test[ix:ix+1]

    plt.figure(figsize=(16, 8))
    plt.subplot(131); plt.title("Low Resolution"); plt.imshow(src[0])
    plt.subplot(132); plt.title("Super-Resolved"); plt.imshow(gen[0])
    plt.subplot(133); plt.title("Original High Resolution"); plt.imshow(tar[0])
    plt.show()
