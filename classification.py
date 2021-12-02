import os
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
import random
import skimage.transform as skt
import skimage.exposure as ske
import skimage.util as sku
import matplotlib.pyplot as plt
from typing import Union, Tuple, List


def read_metadata(df_path):
    df = pd.read_csv(df_path)

    if 'filename' not in df.columns and 'fname' not in df.columns:
        raise ValueError(
            "Missing column contain name file in dataframe. Dataframe must have 'fname' or 'filename' columns.")
    elif 'filename' in df.columns:
        df = df.set_index(keys='filename')
        index = 'filename'
    else:
        df = df.set_index(keys='fname')
        index = 'fname'

    return df


def augment_image(image, vflip: bool = False, hflip: bool = False, rotate_degree: Union[List, Tuple] = None,
                  crop_size: tuple = None, brightness: bool = False, gau_noise: bool = False):
    """
    This function use to apply augmentation on image. There are:
        - Vertical Flip
        - Horizontal Flip
        - Random Rotation (both clockwise and anticlockwise) in range [x,y) degree (losing data)
        - Random crop
        - Random adjust brightness (both lighter and darker)
        - Random add gaussian noise

    :param image: numpy array, shape = (h,w,c)
    :param rotate_degree: (tuple) [x,y) is the range of random degree
    :param crop_size: (tuple) (h,w) size of cropped image (random crop)
    :return: dictionary contain all images after apply augmentation on its
    """
    aug_imgs = {}
    if hflip is True:
        aug_imgs['vflip'] = cv2.flip(image, 0)

    if vflip is True:
        aug_imgs['hflip'] = cv2.flip(image, 1)

    if rotate_degree is not None:
        angle = random.randrange(rotate_degree[0], rotate_degree[1])
        anti_angle = -random.randrange(rotate_degree[0], rotate_degree[1])
        aug_imgs['rotate'] = skt.rotate(image, angle=angle) * 255
        aug_imgs['anti-rotate'] = skt.rotate(image, angle=anti_angle) * 255

    if crop_size is not None:
        crop_size = (min(crop_size[0], image.shape[0]), min(crop_size[1], image.shape[1]))
        start_x = random.randrange(0, image.shape[1] - crop_size[1] + 1)
        start_y = random.randrange(0, image.shape[0] - crop_size[0] + 1)
        end_x = start_x + crop_size[1]
        end_y = start_y + crop_size[0]
        aug_imgs['crop'] = image[start_y:end_y, start_x:end_x, :]

    if brightness is not None:
        gamma_bright = random.randrange(3, 5) / 10
        gamma_dark = random.randrange(21, 23) / 10
        aug_imgs['brighter'] = ske.adjust_gamma(image, gamma=gamma_bright)
        aug_imgs['darker'] = ske.adjust_gamma(image, gamma=gamma_dark)

    if gau_noise is not None:
        aug_imgs['noise'] = sku.random_noise(image) * 255

    return aug_imgs


def augment_data_classification(img_dir, df_path, **kwargs):
    """
    This will automatic add augmentation image in to image directory and add label of those new images to dataframe.
    """
    df = read_metadata(df_path)

    for fname in df.index:
        img_name = '.'.join(fname.split('.')[:-1])
        img_type = fname.split('.')[-1]
        image_path = os.path.join(img_dir, fname)
        image = cv2.imread(image_path)
        aug_imgs = augment_image(image, **kwargs)

        for aug_type, aug_img in aug_imgs.items():
            aug_fname = img_name + '_' + aug_type + '.' + img_type
            # add new image
            cv2.imwrite(os.path.join(img_dir, aug_fname), aug_img)
            # add new row in dataframe
            df.loc[aug_fname] = df.loc[fname]

    df.to_csv(df_path)


def plot_image(img_path):
    image = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    print()


def relabel(img_dir, df_path, label_cols=None, label_nan=False):
    if type(label_cols) is not list or type(label_cols) is not tuple:
        label_cols = [label_cols]
    print(label_cols)

    df = read_metadata(df_path)
    for id, fname in enumerate(df.index):
        if label_nan and all(not np.isnan(df.loc[fname, label_col]) for label_col in label_cols):
            continue

        print('Old Label :', df.loc[fname])
        plot_image(os.path.join(img_dir, fname))
        for label_col in label_cols:
            check = True
            while check:
                try:
                    new_label = float(input('New Label for ' + label_col + ': '))
                    check = False
                except:
                    df.to_csv(df_path)
        df.loc[fname, label_col] = new_label
        print('New Label :', df.loc[fname])
    df.to_csv(df_path)


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help="Metadata path (csv file)",
                        default=r'D:\Machine Learning Project\5kCompliance\dataset\train\train_meta.csv')
    parser.add_argument('--img_dir', type=str, help="Directory path contain images",
                        default=r'D:\Machine Learning Project\5kCompliance\dataset\train\images')
    parser.add_argument('--config', type=str, help="Config path",
                        default=os.path.join(os.getcwd(), 'config.yaml'))
    parser.add_argument('--mode', type=str, help="What do you want to do ? (augmentation or relabel)",
                        default="augmentation")

    return parser.parse_args()


def read_config(config_path, mode):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
    return config_data[mode]


if __name__ == '__main__':
    args = parser_argument()

    img_dir = args.img_dir
    df_path = args.metadata
    kwags = read_config(args.config, args.mode)
    augment_data_classification(img_dir, df_path, **kwags)

    # relabel(img_dir=r'D:\Machine Learning Project\5kCompliance\dataset\train\images',
    #         df_path=r'D:\Machine Learning Project\5kCompliance\dataset\train\train_meta.csv',
    #         label_cols='distancing', label_nan=True)
