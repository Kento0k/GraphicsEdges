import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Отрисовка и сохранение изображения
def draw_image(img, saving_folder, saving_name):
    plt.tick_params(labelsize=0, length=0)
    plt.imshow(img, cmap='gray')
    Path(saving_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(saving_folder + saving_name, bbox_inches='tight', pad_inches=0)
plt.show()

# Сравнение изображений
def img_comparison(first_img, second_img):
    img_height, img_width = first_img.shape
    new_img = np.zeros((img_height, img_width))

    for i in range(0, img_height):
        for j in range(0, img_width):
            new_img[i][j] = abs(first_img[i][j] - second_img[i][j])

    return new_img

def main_func():

    # Чтение изображений
    original_img = cv2.imread('pics/pic1.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
    draw_image(original_img, 'out', '/original_img.png')

    previtt_img = cv2.imread('out/previtt_out/previtt.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
    sobel_img = cv2.imread('out/sobel_out/sobel.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)
    roberts_img = cv2.imread('out/roberts_out/roberts.png', cv2.IMREAD_GRAYSCALE).astype(np.int32)

    # sobel vs previtt
    sobel_vs_previtt = img_comparison(sobel_img, previtt_img)
    draw_image(sobel_vs_previtt, 'out/', 'sobel_vs_previtt.png')

    # sobel vs roberts
    sobel_vs_roberts = img_comparison(sobel_img, roberts_img)
    draw_image(sobel_vs_roberts, 'out/', 'sobel_vs_roberts.png')

    # roberts vs previtt
    roberts_vs_previtt = img_comparison(roberts_img, previtt_img)
    draw_image(roberts_vs_previtt, 'out/', 'roberts_vs_previtt.png')

if __name__ == '__main__':
    main_func()
