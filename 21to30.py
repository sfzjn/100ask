import sys

import cv2
import numpy as np

sys.path.append("util")
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
    x = np.tile(np.arange(aW), (aH, 1))  # tile：把数组沿各个方向复制。先沿x方向复制一倍，再沿y方向复制aH倍
    y = np.round(y / ay).astype(int)  # np.round四舍五入
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

    print("------模仿官方算法------")
    x = np.arange(3).repeat(3, axis=0).reshape(3, 3)
    y = np.arange(3).reshape(1, 3).repeat(3, axis=0)
    x = np.round(x / a).astype(int)
    y = np.round(y / a).astype(int)

    out = img[x, y]


# 双线性插值（ Bilinear Interpolation ）
# Bi-Linear interpolation
def bl_interpolate(img, ax=1., ay=1.):
    H, W, C = img.shape

    aH = int(ay * H)
    aW = int(ax * W)

    # get position of resized image
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    # get position of original position
    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(int)  # np.floor()函数用于以元素方式返回输入的下限,和int()作用应该一样
    iy = np.floor(y).astype(int)

    ix = np.minimum(ix, W - 2)  # 超过W-2设为W-2
    iy = np.minimum(iy, H - 2)

    # get distance
    dx = x - ix
    dy = y - iy

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

    # interpolation
    out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[
        iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def ask26(img):
    a = 1.5

    tm = TimeCost()
    h, w, c = img.shape
    ah = int(h * a)
    aw = int(w * a)

    img1 = np.zeros((ah, aw, c)).astype(float)

    for x in range(ah):
        for y in range(aw):
            x1, y1 = x / a, y / a
            x2, y2 = int(x1), int(y1)
            dx = x1 - x2
            dy = y1 - y2
            img1[x, y] = (1 - dx) * (1 - dy) * img[x2, y2]
            if x2 + 1 < h:
                img1[x, y] += dx * (1 - dy) * img[x2 + 1, y2]
            if y2 + 1 < w:
                img1[x, y] += (1 - dx) * dy * img[x2, y2 + 1]
            if (x2 + 1 < h) & (y2 + 1 < w):
                img1[x, y] += dx * dy * img[x2 + 1, y2 + 1]

    img1 = img1.astype(np.uint8)
    print("耗时:{}s".format(tm.getCostTime()))

    tm = TimeCost()
    img2 = bl_interpolate(img, 1.5, 1.5)
    print("官方耗时:{}s".format(tm.getCostTime()))

    cv_show("compare", np.hstack((img1, img2)))


# 双三次插值（ Bicubic Interpolation ）
def ask27(img):
    print("11")


# 仿射变换（ Afine Transformations ）——平行移动
# Affine
def affine(_img, a, b, c, d, tx, ty):
    H, W, C = _img.shape

    # temporary image
    img = np.zeros((H + 2, W + 2, C), dtype=np.float32)
    img[1:H + 1, 1:W + 1] = _img

    # get new image shape
    H_new = np.round(H * d).astype(int)
    W_new = np.round(W * a).astype(int)
    out = np.zeros((H_new + 1, W_new + 1, C), dtype=np.float32)

    # get position of new image
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    # get position of original image by affine
    adbc = a * d - b * c
    x = np.round((d * x_new - b * y_new) / adbc).astype(int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W + 1).astype(int)
    y = np.minimum(np.maximum(y, 0), H + 1).astype(int)

    # assgin pixcel to new image
    out[y_new, x_new] = img[y, x]

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)
    cv_show("out",out)
    return out


def ask28(img):
    tx = 30
    ty = -30

    tm=TimeCost()
    h, w, c = img.shape
    roi = img[max(0, tx):min(h, h + tx), max(0, ty):min(w, w + ty)]
    cv_show("roi", roi)

    img1 = np.zeros_like(img)
    img1[max(0, 0 - tx):min(h, h - tx), max(0, 0 - ty):min(w, w - ty)] = roi

    print(tm.getCostTime())
    # x=np.arange(h).reshape(h,1)
    # x=np.repeat(x,w,axis=1)
    # y=np.arange(w).reshape(1,w)
    # y=np.repeat(y,h,axis=0)
    #
    # new_x=np.maximum(x+tx,h-1)
    # new_y=np.maximum(y+ty,w-1)
    #
    # img1=img[new_x,new_y]

    cv_show("img1", img1)

    print("-----下面为官方算法-----")
    tm=TimeCost()
    out = affine(img, a=1, b=0, c=0, d=1, tx=30, ty=-30)
    print(tm.getCostTime())


if __name__ == "__main__":
    img = cv2.imread("imori_dark.jpg")
    img1 = cv2.imread("imori.jpg")
    img2 = cv2.imread("dim1.jpg")
    img3 = cv2.imread("imori_gamma.jpg")
    ask28(img1)
