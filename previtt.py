import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Получение маски ky
def get_ky_mask():
    return [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

# Получение маски kx
def get_kx_mask():
    return [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

# Расширение изображения нулями
def expand_img_with_zeros(img):
    return np.pad(img, 1, mode='constant')

# Применение маски к изображению
def img_convolution(img, mask):
    new_img = img.copy()
    img_height, img_width = new_img.shape

    for i in range(1, img_height - 2):
        for j in range(1, img_width - 2):
            new_img[i][j] = abs(img[i-1][j-1] * mask[0][0] +
                                img[i-1][j] * mask[0][1] +
                                img[i-1][j+1] * mask[0][2] +
                                img[i][j-1] * mask[1][0] +
                                img[i][j] * mask[1][1] +
                                img[i][j+1] * mask[1][2] +
                                img[i+1][j-1] * mask[2][0] +
                                img[i+1][j] * mask[2][1] +
                                img[i+1][j+1] * mask[2][2])

    return new_img

# Попиксельное сложение изображений
def img_addition(first_img, second_img):
    new_img = first_img.copy()
    img_height, img_width = first_img.shape

    for i in range(0, img_height):
        for j in range(0, img_width):
            new_img[i][j] = pow(pow(first_img[i][j], 2) + pow(second_img[i][j], 2), 0.5)

    return new_img

# Отрисовка и сохранение изображения
def draw_image(img, saving_folder, saving_name):
    plt.tick_params(labelsize=0, length=0)
    plt.imshow(img, cmap='gray')
    Path(saving_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(saving_folder + saving_name, bbox_inches='tight', pad_inches=0)
    plt.show()

def main_func():

    # Чтение изображения
    img = cv2.imread('pics/pic1.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)

    # Расширение изображения нулями
    expanded_img = expand_img_with_zeros(img)

    # Применение kx
    img_with_kx_mask = img_convolution(expanded_img, get_kx_mask())
    draw_image(img_with_kx_mask, './out/previtt_out', '/previtt_kx.png')

    # Применение ky
    img_with_ky_mask = img_convolution(expanded_img, get_ky_mask())
    draw_image(img_with_ky_mask, './out/previtt_out', '/previtt_ky.png')

    # Сложение результатов проходов с kx и ky
    addition_img = img_addition(img_with_kx_mask, img_with_ky_mask)
    draw_image(addition_img, './out/previtt_out', '/previtt.png')

if __name__ == '__main__':
    main_func()
