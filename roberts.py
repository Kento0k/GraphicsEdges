import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Получение маски k1
def get_k1_mask():
    return [[1, 0], [0, -1]]

# Получение маски k2
def get_k2_mask():
    return [[0, 1], [-1, 0]]

# Расширение изображения нулями
def expand_img_with_zeros(img):
    return np.pad(img, 1, mode='constant')

# Применение маски к изображению
def img_convolution(img, mask):
    new_img = img.copy()
    img_height, img_width = new_img.shape

    for i in range(1, img_height - 2):
        for j in range(1, img_width - 2):
            new_img[i][j] = abs(img[i][j] * mask[0][0] +
                                img[i][j+1] * mask[0][1] +
                                img[i+1][j] * mask[1][0] +
                                img[i+1][j+1] * mask[1][1])

    return new_img

# Попиксельное сложение изображений
def img_addition(first_img, second_img):
    new_img = first_img.copy()
    img_height, img_width = first_img.shape

    for i in range(0, img_height):
        for j in range(0, img_width):
            new_img[i][j] = first_img[i][j] + second_img[i][j]

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

    # Применение k1
    img_with_k1_mask = img_convolution(expanded_img, get_k1_mask())
    draw_image(img_with_k1_mask, './out/roberts_out', '/roberts_k1.png')

    # Применение k2
    img_with_k2_mask = img_convolution(expanded_img, get_k2_mask())
    draw_image(img_with_k2_mask, './out/roberts_out', '/roberts_k2.png')

    # Сложение результатов проходов с k1 и k2
    addition_img = img_addition(img_with_k1_mask, img_with_k2_mask)
    draw_image(addition_img, './out/roberts_out', '/roberts.png')

if __name__ == '__main__':
    main_func()
