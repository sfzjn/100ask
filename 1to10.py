import cv2
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


# 通道交换
def ask1(img):
    b, g, r = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = r, g, b
    cv_show("ask1", img)

    img = img[:, :, (2, 1, 0)]
    cv_show("ask11", img)


# 灰度化
def ask2(img):
    h, w, c = img.shape
    # grayimg=np.zeros((h,w,1),np.uint8)
    b, g, r = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    grayimg = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
    print(grayimg)
    print(type(grayimg[0, 0]))

    grayimg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show("gray compare", np.hstack((grayimg, grayimg1)))

    return grayimg

# 二值化
def ask3(img):
    grayimg=ask2(img)

    h,w=grayimg.shape
    th1=grayimg.copy()
    for x in range(h):
        for y in range(w):
            th1[x,y]=0 if grayimg[x,y]<128 else 255
    #cv_show("Threshold",th1)

    th2=grayimg.copy()
    th2[th2<128]=0
    th2[th2>=128]=255

    th3=np.where(grayimg<128,0,255).astype(np.uint8)
    print(type(th3[0,0]))
    cv_show("Threshold",np.hstack((th1,th2,th3)))

#大津二值化算法
def ask4(img):


if __name__ == "__main__":
    img = cv2.imread("imori.jpg")
    # ask1(img)
    ask3(img)
