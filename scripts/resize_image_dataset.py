#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


"""
This is an example of the kind of script you'd need.
This script resizes images in a folder using crop and pad; adapt to your dataset.
"""

import os
import shutil
import sys

import face_recognition as fr
from PIL import Image

sys.path.append("vendor")
import cv2
import numpy as np

_root = "data/smileys"
size = 348
final_size = 128
# crop_sz = 128


def face_alignment(pil_img, scale=0.9, face_size=(224, 224), target_eye_distance=300):
    """
    face alignment API for single image, get the landmark of eyes and nose and do warpaffine transformation
    :param face_img: single image that including face, I recommend to use dlib frontal face detector
    :param scale: scale factor to judge the output image size

    :return: an aligned single face image
    """
    img = np.asarray(pil_img)
    h, w, c = img.shape
    face_loc = _face_locations_small(img)[0]
    # img_draw = ImageDraw.Draw(pil_img)
    # img_draw.rectangle((face_loc[1], face_loc[0], face_loc[3], face_loc[2]), outline='red', width=5)
    # pil_img.show()
    face_img = _crop_face(
        img, face_loc
    )  # , padding_size=int((face_loc[2] - face_loc[0])*0.5))
    # pil_img = Image.fromarray(face_img)
    face_loc_small_img = _face_locations_small(face_img)
    # drawing = ImageDraw.Draw(pil_img)
    # drawing.rectangle((face_loc_small_img[0][1], face_loc_small_img[0][0], face_loc_small_img[0][3], face_loc_small_img[0][2]), outline='red', width=5)
    # pil_img.show()
    face_land = fr.face_landmarks(face_img, face_loc_small_img)
    if len(face_land) == 0:
        return []
    left_eye_center = _find_center_pt(face_land[0]["left_eye"])
    right_eye_center = _find_center_pt(face_land[0]["right_eye"])
    nose_center = _find_center_pt(face_land[0]["nose_tip"])
    trotate = _get_rotation_matrix(
        left_eye_center, right_eye_center, nose_center, img, scale=scale
    )
    # Warp the original image so that it's centered on the nose and rescaled.
    warped = cv2.warpAffine(img, trotate, (w, h))
    # print("Warped original image")
    # pil_img = Image.fromarray(warped)
    # pil_img.show()
    new_locations = _face_locations_small(warped)
    new_landmarks = fr.face_landmarks(warped, new_locations)
    left_eye_center = _find_center_pt(new_landmarks[0]["left_eye"])
    right_eye_center = _find_center_pt(new_landmarks[0]["right_eye"])
    if len(new_landmarks) == 0:
        return []
    # output_img= _crop_face(
    # warped, new_locations[0], padding_size=int((new_locations[0][2] -
    # new_locations[0][0])*0.35)
    # )
    # We want the new width to be the old one minus the distance between the eyes times a scalar,
    # and the new height to be proportional to the new width.
    eye_dist = right_eye_center[0] - left_eye_center[0]
    # print(f"Original eye distance: {eye_dist}")
    # print(f"Target eye distance: {target_eye_distance}")
    new_width = int(target_eye_distance * w / eye_dist)
    new_height = int(new_width * h / w)
    old_h, old_w = h, w
    # print("Old: ", old_h, old_w)
    # print("New: ", new_height, new_width)
    warped = cv2.resize(warped, (new_width, new_height))
    h, w, c = warped.shape
    assert h == new_height
    assert w == new_width
    # Crop the image such that the eyes are centered somewhere.
    output_img = warped[
        (left_eye_center[1] * new_height // old_h)
        - int(0.35 * new_height) : (right_eye_center[1] * new_height // old_h)
        + int(0.35 * new_height),
        (left_eye_center[0] * new_width // old_w)
        - int(0.3 * new_width) : (right_eye_center[0] * new_width // old_w)
        + int(0.3 * new_width),
    ]
    # new_locations = _face_locations_small(output_img)
    # new_landmarks = fr.face_landmarks(output_img, new_locations)
    # left_eye_center = _find_center_pt(new_landmarks[0]["left_eye"])
    # right_eye_center = _find_center_pt(new_landmarks[0]["right_eye"])
    # print(f" New eye distance: {(right_eye_center[0] - left_eye_center[0])}")
    return output_img


def _find_center_pt(points):
    """
    find centroid point by several points that given
    """
    x = 0
    y = 0
    num = len(points)
    for pt in points:
        x += pt[0]
        y += pt[1]
    x //= num
    y //= num
    return (x, y)


def _angle_between_2_pt(p1, p2):
    """
    to calculate the angle rad by two points
    """
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan_angle))


def _get_rotation_matrix(left_eye_pt, right_eye_pt, nose_center, face_img, scale):
    """
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    """
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    print(f"eye angle: {eye_angle}")
    # Rotate around the nose by the eye angle and rescale by scale.
    # TODO: Shouldn't the scale be the normalized distance between the eyes?
    M = cv2.getRotationMatrix2D(
        (nose_center[0] / 2, nose_center[1] / 2), eye_angle, scale
    )

    return M


def _dist_nose_tip_center_and_img_center(nose_pt, img_shape):
    """
    find the distance between nose tip's centroid and the centroid of original image
    """
    y_img, x_img, _ = img_shape
    img_center = (x_img // 2, y_img // 2)
    return ((img_center[0] - nose_pt[0]), -(img_center[1] - nose_pt[1]))


def _crop_face(img, face_loc, padding_size=0):
    """
    crop face into small image, face only, but the size is not the same
    """
    h, w, c = img.shape
    top = face_loc[0] - padding_size
    right = face_loc[1] + padding_size
    down = face_loc[2] + padding_size
    left = face_loc[3] - padding_size

    if top < 0:
        top = 0
    if right > w - 1:
        right = w - 1
    if down > h - 1:
        down = h - 1
    if left < 0:
        left = 0
    img_crop = img[top:down, left:right]
    return img_crop


def _crop_face_2(img, left_eye_center, right_eye_center, padding_size=(0, 0)):
    h, w, c = img.shape
    top = left_eye_center[1] - padding_size[0]
    left = left_eye_center[0] - padding_size[1]

    if top < 0:
        top = 0
    # if right > w - 1:
    # right = w - 1
    # if down > h - 1:
    #     down = h - 1
    if left < 0:
        left = 0
    img_crop = img[top:, left:]
    return img_crop


def _face_locations_raw(img, scale):
    h, w, c = img.shape
    img_scale = cv2.resize(
        img, (int(img.shape[1] // scale), int(img.shape[0] // scale))
    )
    # print("Scale factor: ", scale)
    # pil_img = Image.fromarray(img_scale)
    # drawing = ImageDraw.Draw(pil_img)
    face_loc_small = fr.face_locations(img_scale, model="hog")
    # print(face_loc_small)
    # drawing.rectangle((face_loc_small[0][1], face_loc_small[0][0], face_loc_small[0][3],
    # face_loc_small[0][2]), outline="red")
    # pil_img.show()
    face_loc = []
    for ff in face_loc_small:
        tmp = [pt * scale for pt in ff]
        if tmp[1] >= w:
            tmp[1] = w
        if tmp[2] >= h:
            tmp[2] = h
        face_loc.append(tmp)
    return face_loc


def _face_locations_small(img):
    for scale in [16, 8, 4, 2, 1]:
        face_loc = _face_locations_raw(img, scale)
        if face_loc != []:
            return face_loc
    return []


for root, dirs, files in os.walk(_root):
    for file in files:
        img_path = os.path.join(root, file)
        while img_path.endswith(".old"):
            shutil.move(img_path, f"{img_path[:-4]}")
            img_path = f"{img_path[:-4]}"
        moved_path = os.path.join(root, file + ".old")
        shutil.move(img_path, moved_path)
        img = Image.open(moved_path)
        w, h = img.size
        if h == 348:
            continue
        new_size = size, size
        if w > h:
            new_size = (size, h * size // w)
        elif h > w:
            new_size = (w * size // h, size)
        new_img = img.resize(new_size)
        # width, height = new_img.size  # Get dimensions
        # left = (width - crop_sz) // 2
        # top = (height - crop_sz) // 2
        # right = (width + crop_sz) // 2
        # bottom = (height + crop_sz) // 2

        # Crop the center of the image
        # new_img = new_img.crop((left, top, right, bottom))
        new_img = img.rotate(180)
        # plt.imshow(new_img)
        # plt.show()
        aligned = face_alignment(new_img, scale=1.00)
        # final = Image.fromarray(aligned[0])
        # w, h = final.size
        # crop_size = 2100
        # final = final.crop((0, 600, 0 + crop_size, 600 + crop_size))
        # final = final.resize((final_size, final_size))
        # plt.imshow(np.asarray(final))
        # plt.show()
        # final.save(img_path)
        Image.fromarray(aligned).save(img_path)
        img.close()
        new_img.close()
        os.remove(moved_path)
