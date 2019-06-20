import numpy as np
import math
import cv2
import os


def PSNR(orig, reconstr):
    mse = np.mean((orig.astype(float) - reconstr.astype(float)) ** 2)
    if mse != 0:
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))
    else:
        return 1


##for feeding tf.data
def gen_dataset(filenames, scale):
    # The model trains on 17x17 patches
    crop_size_lr = 17
    crop_size_hr = 17 * scale

    # x = list()
    # y = list()
    for p in filenames:
        image_decoded = cv2.imread(p.decode(), 3).astype(np.float32) / 255.0
        imgYCC = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCrCb)
        cropped = imgYCC[0:(imgYCC.shape[0] - (imgYCC.shape[0] % scale)),
                  0:(imgYCC.shape[1] - (imgYCC.shape[1] % scale)), :]
        lr = cv2.resize(cropped, (int(cropped.shape[1] / scale), int(cropped.shape[0] / scale)),
                        interpolation=cv2.INTER_CUBIC)

        hr_y = imgYCC[:, :, 0]
        lr_y = lr[:, :, 0]

        numx = int(lr.shape[0] / crop_size_lr)
        numy = int(lr.shape[1] / crop_size_lr)
        for i in range(0, numx):
            startx = i * crop_size_lr
            endx = (i * crop_size_lr) + crop_size_lr
            startx_hr = i * crop_size_hr
            endx_hr = (i * crop_size_hr) + crop_size_hr
            for j in range(0, numy):
                starty = j * crop_size_lr
                endy = (j * crop_size_lr) + crop_size_lr
                starty_hr = j * crop_size_hr
                endy_hr = (j * crop_size_hr) + crop_size_hr

                crop_lr = lr_y[startx:endx, starty:endy]
                crop_hr = hr_y[startx_hr:endx_hr, starty_hr:endy_hr]

                hr = crop_hr.reshape((crop_size_hr, crop_size_hr, 1))
                lr = crop_lr.reshape((crop_size_lr, crop_size_lr, 1))
                yield lr, hr

                # x.append(lr)
                # y.append(hr)

    # X = x
    # Y = y
    # return X, Y
