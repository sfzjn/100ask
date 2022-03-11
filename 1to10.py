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
    grayimg = ask2(img)

    h, w = grayimg.shape
    th1 = grayimg.copy()
    for x in range(h):
        for y in range(w):
            th1[x, y] = 0 if grayimg[x, y] < 128 else 255
    # cv_show("Threshold",th1)

    th2 = grayimg.copy()
    th2[th2 < 128] = 0
    th2[th2 >= 128] = 255

    th3 = np.where(grayimg < 128, 0, 255).astype(np.uint8)
    print(type(th3[0, 0]))
    cv_show("Threshold", np.hstack((th1, th2, th3)))


# 大津二值化算法
def ask4(img):
    th = 128
    max_sigma = 0
    max_t = 0

    grayimg = ask2(img)
    out = grayimg.copy()
    list = []
    h, w = grayimg.shape
    max1, max_index = 0, 0
    for i in range(1, 255):
        t = grayimg[np.where(grayimg < i)]
        t1 = grayimg[grayimg < i]
        m0 = np.mean(t) if len(t) > 0 else 0
        w0 = t.size / grayimg.size

        t = grayimg[np.where(grayimg >= i)]
        m1 = np.mean(t) if len(t) > 0 else 0
        w1 = t.size / grayimg.size

        tmp_max = w0 * w1 * (m0 - m1) ** 2
        if tmp_max > max1:
            max1 = tmp_max
            max_index = i

    print("my:", max_index)

    # determine threshold
    H, W, C = img.shape
    for _t in range(1, 255):
        tmp1 = np.where(out < _t)
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255
    cv_show("out", out)


# HSV变换
def BGR2HSV(_img):
    img = _img.copy() / 255.

    hsv = np.zeros_like(img, dtype=np.float32)

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    # H
    hsv[..., 0][np.where(max_v == min_v)] = 0
    ## if min == B
    ind = np.where(min_arg == 0)
    # ind是最小值为B值的索引，两个元组分别是x,y坐标
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## if min == R
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## if min == G
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

    # S
    hsv[..., 1] = max_v.copy() - min_v.copy()

    # V
    hsv[..., 2] = max_v.copy()

    return hsv


def HSV2BGR(_img, hsv):
    img = _img.copy() / 255.

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    out = np.zeros_like(img)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs(H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i + 1)))
        out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

    out[np.where(max_v == min_v)] = 0
    #指定out上下界为0,1
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out


def ask5(img):
    originimg = img.copy()

    t1=cv2.getTickCount()
    h, w, c = img.shape
    hsv1 = np.zeros_like(img)
    for x in range(h):
        for y in range(w):
            # print(x, y)
            # max_c = max(img[x, y])
            # min_c = min(img[x, y])
            B, G, R = int(img[x, y, 0]), int(img[x, y, 1]), int(img[x, y, 2])
            B, G, R = B / 255, G / 255, R / 255
            max_c = max(B, G, R)
            min_c = min(B, G, R)
            tmp = max_c - min_c
            H = 0
            if min_c == max_c:
                H = 0
            elif min_c == B:
                H = 60 * (G - R) / tmp + 60
            elif min_c == R:
                H = 60 * (B - G) / tmp + 180
            elif min_c == G:
                H = 60 * (R - B) / tmp + 300

            S = max_c - min_c
            V = max_c
            H = (H+180) % 360
            # print(H)
            hsv1[x, y] = [H, S, V]

            C = S
            H1 = H / 60
            X = C * (1 - abs(H1 % 2 - 1))
            tmp = [0, 0, 0]
            if H1 < 1 and H1 >= 0:
                tmp = [C, X, 0]
            elif H1 < 2 and H1 >= 1:
                tmp = [X, C, 0]
            elif H1 < 3 and H1 >= 2:
                tmp = [0, C, X]
            elif H1 < 4 and H1 >= 3:
                tmp = [0, X, C]
            elif H1 < 5 and H1 >= 4:
                tmp = [X, 0, C]
            elif H1 < 6 and H1 >= 5:
                tmp = [C, 0, X]

            img[x, y] = ((V - C) * np.array([1, 1, 1]) + tmp) * np.array([255, 255, 255])
            #print((V - C) * np.array([1, 1, 1]))
            #print("tmp:",tmp)

    #print(img)
    img = img[:, :, (2, 1, 0)].astype("uint8")

    t2=cv2.getTickCount()
    print("我的耗时：%rus"%((t2-t1)/cv2.getTickFrequency()))
    #print("还原:",img)
    #cv2.imwrite("r.jpg", img)
    #cv_show("result", img)

    print("-----官方运算过程-----")
    t1=cv2.getTickCount()
    hsv = BGR2HSV(originimg)
    # print("hsv:", hsv)
    # print("hsv1:", hsv1)
    # print(hsv1 == hsv)
    hsv[..., 0] = (hsv[..., 0] + 180) % 360
    out = HSV2BGR(originimg, hsv)
    t2=cv2.getTickCount()
    print("官方耗时：%rus"%((t2-t1)/cv2.getTickFrequency()))
    cv_show("compare",np.hstack((img,out)))

    # print(out)
    #cv_show("out", out)


if __name__ == "__main__":
    img = cv2.imread("imori.jpg")
    # ask1(img)
    #img = np.random.randint(0, 255, (1, 1, 3)).astype("uint8")
    # =np.vstack((img,img))
    # img=np.hstack((img,img))
    print("img:",img)
    # cv_show("img",img)
    ask5(img)
