import os
import cv2
import numpy as np
import pandas as pd
import random
import skimage.transform as skt
import skimage.exposure as ske
import skimage.util as sku


def augment_image(image, vflip: bool = False, hflip: bool = False, rotate: tuple = None, crop: bool = True,
                  brightness: bool = False, gau_noise: bool = False):
    """
    This function use to apply augmentation on image. There are:
        - Vertical Flip
        - Horizontal Flip
        - Random Rotation (both clockwise and anticlockwise) in range [x,y) degree (losing data)
        - Random crop
        - Random adjust brightness (both lighter and darker)
        - Random add gaussian noise

    :param image: numpy array
    :param rotate: (tuple) [x,y) that is the range of degree to rotate
    :return: dictionary contain all images after apply augmentation on its
    """
    aug_imgs = {}
    if hflip is True:
        aug_imgs['vflip'] = cv2.flip(image, 0)

    if vflip is True:
        aug_imgs['hflip'] = cv2.flip(image, 1)

    if rotate is not None:
        angle = random.randrange(rotate[0], rotate[1])
        anti_angle = -random.randrange(rotate[0], rotate[1])
        aug_imgs['rotate'] = skt.rotate(image, angle=angle) * 255
        aug_imgs['anti-rotate'] = skt.rotate(image, angle=anti_angle) * 255

    if crop is not None:
        pass

    if brightness is not None:
        gamma_bright = random.randrange(3, 5) / 10
        gamma_dark = random.randrange(20, 22) / 10
        aug_imgs['brighter'] = ske.adjust_gamma(image, gamma=gamma_bright)
        aug_imgs['darker'] = ske.adjust_gamma(image, gamma=gamma_dark)

    if gau_noise is not None:
        aug_imgs['noise'] = sku.random_noise(image) * 255

    return aug_imgs


def augment_data_classification(img_dir, df_path, vflip: bool = False, hflip: bool = False, rotate: tuple = None,
                                crop: bool = True, brightness: bool = False, gau_noise: bool = False):
    df = pd.read_csv(df_path, index_col=0)

    if 'filename' not in df.columns and 'fname' not in df.columns:
        raise ValueError(
            "Missing column contain name file in dataframe. Dataframe must have 'fname' or 'filename' columns.")
    elif 'filename' in df.columns:
        df = df.set_index(keys='filename')
        index = 'filename'
    else:
        df = df.set_index(keys='fname')
        index = 'fname'

    for fname in df.index:
        img_name = fname.split('.')[0]
        img_type = fname.split('.')[-1]
        image_path = os.path.join(img_dir, fname)
        image = cv2.imread(image_path)
        aug_imgs = augment_image(image, vflip=vflip, hflip=hflip, rotate=rotate, crop=crop, brightness=brightness,
                                 gau_noise=gau_noise)

        for aug_type, aug_img in aug_imgs.items():
            aug_fname = img_name + '_' + aug_type + '.' + img_type
            cv2.imwrite(os.path.join(img_dir, aug_fname), aug_img)
            df.loc[aug_fname] = df.loc[fname]

    df.to_csv(df_path)
