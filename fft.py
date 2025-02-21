import numpy as np
import cv2

def apply_fft(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    return f_transform_shifted

def inverse_fft(f_transform_shifted):
    f_ishift = np.fft.ifftshift(f_transform_shifted)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

def apply_fft_contrast(image, factor=1.5):
    f_transform_shifted = apply_fft(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    d = 30
    mask = np.ones((rows, cols), np.float32)
    mask[crow - d:crow + d, ccol - d:ccol + d] = factor
    f_transform_shifted *= mask
    return inverse_fft(f_transform_shifted)

def apply_fft_rotation(image, angle):
    f_transform_shifted = apply_fft(image)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    rotated_fft = cv2.warpAffine(np.log(1 + np.abs(f_transform_shifted)), M, (cols, rows))
    return inverse_fft(rotated_fft)

def apply_fft_zoom(image, factor=1.2):
    f_transform_shifted = apply_fft(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    d = int(min(rows, cols) // (2 * factor))
    mask = np.zeros((rows, cols), np.float32)
    mask[crow - d:crow + d, ccol - d:ccol + d] = 1
    f_transform_shifted *= mask
    return inverse_fft(f_transform_shifted)
