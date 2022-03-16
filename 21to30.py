import cv2
import numpy as np

from common import TimeCost
from common import cv_show


# 直方图归一化（ Histogram Normalization ）
def hist_normalization(img, a=0, b=255):
    # get max and min
    c = img.min()
    d = img.max()

    out = img.copy()

    # normalization
    out = (b - a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)

    return out


def ask21(img):
    img1 = img.copy()
    img11 = img.copy()
    h, w, _ = img.shape

    tm = TimeCost()
    for i in range(3):
        max_value = np.max(img[:, :, i])
        min_value = np.min(img[:, :, i])
        weight_value = 255 / (max_value - min_value)
        print(max_value, min_value, weight_value)

        value_map = [None] * 256
        for j in range(256):
            value_map[j] = int(weight_value * (j - min_value))

        for x in range(h):
            for y in range(w):
                img1[x, y, i] = value_map[img[x, y, i]]

    print("方法1耗时：", tm.getCostTime())

    print("------------------------")

    tm = TimeCost()
    max_value = np.max(img[:, :], axis=(0, 1))
    min_value = np.min(img[:, :], axis=(0, 1))
    weight_value = 255 / (max_value - min_value)

    value_map = np.zeros((256, 3)).astype(int)
    for j in range(min(min_value), max(max_value) + 1):
        # print(j)
        value_map[j] = (weight_value * (j - min_value)).astype(int)

    for x in range(h):
        for y in range(w):
            for i in range(3):
                img11[x, y, i] = value_map[img[x, y, i], i]
    print("方法2耗时：", tm.getCostTime())
    cv_show("compare", np.hstack((img, img1, img11)))

    print("------只归一化灰度值------")

    img2 = img.copy()
    img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    max_value = np.max(img_hsv[:, :, 2])
    min_value = np.min(img_hsv[:, :, 2])
    weight_value = 255 / (max_value - min_value)

    value_map = [None] * 255
    for j in range(255):
        value_map[j] = int(weight_value * (j - min_value))

    for x in range(h):
        for y in range(w):
            img_hsv[x, y, 2] = value_map[img_hsv[x, y, 2]]
    img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv_show("img2", img2)

    print("-----官方方法-----")
    tm = TimeCost()
    cv_show("img", hist_normalization(img, 0, 255))
    print("官方耗时：", tm.getCostTime())


# 直方图操作
def ask22(img):
    m0 = 128
    s0 = 52

    m = np.mean(img)
    s = np.std(img)

    img1 = (s0 / s * (img - m) + m0).astype(np.uint8)

    cv_show("img", img1)


# 直方图均衡化
def hist_equal(img, z_max=255):
    H, W, C = img.shape
    S = H * W * C * 1.

    out = img.copy()

    sum_h = 0.

    for i in range(1, 255):
        if i == 64:
            print()
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out


# 直方图均衡化（ Histogram Equalization ）
def ask23(img):
    cv_show("compare", np.hstack((img, hist_equal(img))))


# 伽玛校正（Gamma Correction）
def ask24(img):
    c = 1
    g = 2.2

    img1 = pow(img / 255.0, 1 / g) / c
    img1 = (img1 * 255).astype(np.uint8)
    cv_show("img1", img1)


# 最邻近插值（ Nearest-neighbor Interpolation ）官方算法更简洁
# Nereset Neighbor interpolation
def nn_interpolate(img, ax=1, ay=1):
    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))
    y = np.round(y / ay).astype(int)
    x = np.round(x / ax).astype(int)

    out = img[y, x]

    out = out.astype(np.uint8)

    return out


def ask25(img):
    a = 1.5

    tm = TimeCost()
    h, w, _ = img.shape
    h = int(h * a)
    w = int(w * a)
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            img1[x, y] = img[int(x / a), int(y / a)]

    print("tm:", tm.getCostTime())
    cv_show("img1", img1)

    print("-------官方算法-------")
    tm = TimeCost()
    nn_interpolate(img, 1.5, 1.5)
    print("tm:", tm.getCostTime())


if __name__ == "__main__":
    img = cv2.imread("imori_dark.jpg")
    img1 = cv2.imread("imori.jpg")
    img2 = cv2.imread("dim1.jpg")
    img3 = cv2.imread("imori_gamma.jpg")
    ask25(img1)
