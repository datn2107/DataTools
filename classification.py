import copy
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
from typing import Union, Tuple, List, Optional

ARRAY_LIKE = [list, tuple, np.array]


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


def random_box(image_size, box_size):
    min_x = random.randrange(0, image_size[0] - box_size[0] + 1)
    max_x = min_x + box_size[0]
    min_y = random.randrange(0, image_size[1] - box_size[1] + 1)
    max_y = min_y + box_size[1]
    return min_x, max_x, min_y, max_y


def random_crop_image(image, crop_size):
    image_size = image.shape[0:2]
    crop_size = (min(crop_size[0], image_size[0]), min(crop_size[1], image_size[1]))
    min_x, max_x, min_y, max_y = random_box(image_size, crop_size)
    return image[min_x:max_x, min_y:max_y]


def zoom_image(image, zoom_frac):
    black_img = np.zeros_like(image)
    h, w = image.shape[0], image.shape[1]

    zoomed_img = cv2.resize(image, None, fx=zoom_frac, fy=zoom_frac)
    zh, zw = zoomed_img.shape[0], zoomed_img.shape[1]

    if zoom_frac < 1:
        black_img[(h - zh) // 2:-(h - zh) // 2, (w - zw) // 2:-(w - zw) // 2] = zoomed_img
        return black_img
    else:
        return zoomed_img[(zh - h) // 2:-(zh - h) // 2, (zw - w) // 2:-(zw - w) // 2]


def vertical_shift(image, shift_frac):
    out_img = np.zeros_like(image)
    height_shift = int(image.shape[0] * shift_frac)
    if height_shift > 0:
        out_img[:-height_shift] = image[height_shift:]
    else:
        out_img[-height_shift:] = image[:height_shift]

    return out_img


def horizontal_shift(image, shift_frac):
    out_img = np.zeros_like(image)
    width_shift = int(image.shape[0] * shift_frac)
    if width_shift > 0:
        out_img[:, width_shift:] = image[:, :-width_shift]
    else:
        out_img[:, :width_shift] = image[:, -width_shift:]

    return out_img


def cutout_imgae(image, count, size_frac):
    h, w, _ = image.shape
    ch, cw = int(h * size_frac), int(w * size_frac)
    cutout_image = copy.deepcopy(image)

    for i in range(count):
        min_x, max_x, min_y, max_y = random_box((h, w), (ch, cw))
        cutout_image[min_x:max_x, min_y:max_y] = 0

    return cutout_image


def shear_image(image, degree=0.2):
    afine_tf = skt.AffineTransform(shear=degree)
    modified = skt.warp(image, inverse_map=afine_tf)*255

    return modified


def augment_image(image, vflip=False, hflip=False, rotate_degree=None, shear=None, crop_size=None, cutout=None,
                  random_zoom_in=None, random_zoom_out=None, vshift=None, hshift=None, vhshift=None, brightness=None,
                  gau_noise=False):
    """
    This function use to apply augmentation on image. There are:
        - Vertical Flip
        - Horizontal Flip
        - Random Rotation (both clockwise and anticlockwise) in range [x,y) degree (losing data)
        - Random crop
        - Random adjust brightness (both lighter and darker)
        - Random add gaussian noise
    """
    aug_imgs = {}
    if vflip:
        aug_imgs['vflip'] = cv2.flip(image, 0)

    if hflip:
        aug_imgs['hflip'] = cv2.flip(image, 1)

    if rotate_degree is not None:
        if type(rotate_degree) not in ARRAY_LIKE:
            angle = rotate_degree
            anti_angle = -rotate_degree
        else:
            angle = random.randint(rotate_degree[0], rotate_degree[1])
            anti_angle = -random.randint(rotate_degree[0], rotate_degree[1])

        if random.getrandbits(1) == 0:
            aug_imgs['rotate'] = (skt.rotate(image, angle=angle) * 255).astype(np.uint8)
        else:
            aug_imgs['anti-rotate'] = (skt.rotate(image, angle=anti_angle) * 255).astype(np.uint8)

    if shear is not None:
        if type(shear) not in ARRAY_LIKE:
            shear_degree = shear
        else:
            shear_degree = round(random.uniform(shear[0], shear[1]), 2)
        shear_degree *= random.choice([-1, 1])
        aug_imgs['shear'] = (shear_image(image, degree=shear_degree)).astype(np.uint8)

    if crop_size is not None:
        aug_imgs['crop'] = random_crop_image(image, crop_size)

    if random_zoom_in is not None:
        if type(random_zoom_in) not in ARRAY_LIKE:
            zoom_frac = 1 + random_zoom_in
        else:
            zoom_frac = 1 + round(random.uniform(random_zoom_in[0], random_zoom_in[1]), 2)
        aug_imgs['zoom_in'] = zoom_image(image, zoom_frac)

    if random_zoom_out is not None:
        if type(random_zoom_out) not in ARRAY_LIKE:
            zoom_frac = 1 - random_zoom_out
        else:
            zoom_frac = 1 - round(random.uniform(random_zoom_out[0], random_zoom_out[1]), 2)
        aug_imgs['zoom_out'] = zoom_image(image, zoom_frac)

    if vshift is not None:
        if type(vshift) not in ARRAY_LIKE:
            shift_frac = vshift
        else:
            shift_frac = random.choice((1, -1)) * round(random.uniform(vshift[0], vshift[1]), 2)
        aug_imgs['vshift'] = vertical_shift(image, shift_frac)

    if hshift is not None:
        if type(hshift) not in ARRAY_LIKE:
            shift_frac = hshift
        else:
            shift_frac = random.choice((1, -1)) * round(random.uniform(hshift[0], hshift[1]), 2)
        aug_imgs['hshift'] = horizontal_shift(image, shift_frac)

    if vhshift is not None:
        if type(vhshift[0]) not in ARRAY_LIKE:
            shift_frac_v = vhshift[0]
        else:
            shift_frac_v = random.choice((1, -1)) * round(random.uniform(vhshift[0][0], vhshift[0][1]), 2)
        if type(vhshift[1]) not in ARRAY_LIKE:
            shift_frac_h = vhshift[1]
        else:
            shift_frac_h = random.choice((1, -1)) * round(random.uniform(vhshift[1][0], vhshift[1][1]), 2)
        aug_imgs['vhshift'] = horizontal_shift(vertical_shift(image, shift_frac_v), shift_frac_h)

    if cutout is not None:
        if type(cutout[0]) not in ARRAY_LIKE:
            count = cutout[0]
        else:
            count = random.randint(cutout[0][0], cutout[0][1])
        if type(cutout[1]) not in ARRAY_LIKE:
            size_frac = cutout[1]
        else:
            size_frac = round(random.uniform(cutout[1][0], cutout[1][1]), 2)
        aug_imgs['cutout'] = cutout_imgae(image, count, size_frac)

    if brightness:
        gamma_bright = random.randrange(3, 5) / 10
        gamma_dark = random.randrange(21, 23) / 10
        if brightness == 0 or brightness == 2:
            aug_imgs['brighter'] = ske.adjust_gamma(image, gamma=gamma_bright)
        if brightness == 1 or brightness == 2:
            aug_imgs['darker'] = ske.adjust_gamma(image, gamma=gamma_dark)

    if gau_noise:
        aug_imgs['noise'] = (sku.random_noise(image) * 255).astype(np.uint8)

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


# def relabel(img_dir, df_path, label_cols=None, label_nan=False):
#     if type(label_cols) is not list or type(label_cols) is not tuple:
#         label_cols = [label_cols]
#     print(label_cols)
#
#     df = read_metadata(df_path)
#     for id, fname in enumerate(df.index):
#         if label_nan and all(not np.isnan(df.loc[fname, label_col]) for label_col in label_cols):
#             continue
#
#         print('Old Label :', df.loc[fname])
#         plot_image(os.path.join(img_dir, fname))
#         for label_col in label_cols:
#             check = True
#             while check:
#                 try:
#                     new_label = float(input('New Label for ' + label_col + ': '))
#                     check = False
#                 except:
#                     df.to_csv(df_path)
#         df.loc[fname, label_col] = new_label
#         print('New Label :', df.loc[fname])
#     df.to_csv(df_path)


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help="Metadata path (csv file)",
                        default=r'D:\Machine Learning Project\5kCompliance\dataset\train\train_meta.csv')
    parser.add_argument('--img_dir', type=str, help="Directory path contain images",
                        default=r'D:\Machine Learning Project\5kCompliance\dataset\train\images')
    parser.add_argument('--config', type=str, help="Config path",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml'))
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

    # dataroot = r'D:\Machine Learning Project\5kCompliance\dataset\face_mask\FaceMaskClassification\Face-Mask-Dataset\Train\WithMask'
    # image = random.choice(os.listdir(dataroot))
    # # image = 'facemaskdetection_1_maksssksksss135.png'
    # print(image)
    # image = cv2.imread(
    #     os.path.join(dataroot, image))
    # image = cv2.resize(image, (128, 128))
    # dic = augment_image(image, **kwags)
    # for key, value in dic.items():
    #     print(key)
    #     print(value.shape)
    #     plt.imshow(cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
    #     plt.show()

    # relabel(img_dir=r'D:\Machine Learning Project\5kCompliance\dataset\train\images',
    #         df_path=r'D:\Machine Learning Project\5kCompliance\dataset\train\train_meta.csv',
    #         label_cols='distancing', label_nan=True)
