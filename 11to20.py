import cv2
import numpy as np
from util.common import cv_show


def convolutin(kernel, grayimg):
    h, w = grayimg.shape

    img1 = cv2.copyMakeBorder(grayimg, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    img2 = grayimg.copy()
    img22 = grayimg.copy()
    img3 = grayimg.copy()
    img33 = grayimg.copy()

    for x in range(h):
        for y in range(w):
            img2[x, y] = abs(np.sum(img1[x:x + 3, y:y + 3] * kernel, axis=(0, 1)))
            img3[x, y] = abs(np.sum(img1[x:x + 3, y:y + 3] * kernel.T, axis=(0, 1)))
            img22[x, y] = np.clip((np.sum(img1[x:x + 3, y:y + 3] * kernel, axis=(0, 1))), 0, 255)
            img33[x, y] = np.clip((np.sum(img1[x:x + 3, y:y + 3] * kernel.T, axis=(0, 1))), 0, 255)

    cv_show("compare", np.hstack((grayimg, img2, img22, img3, img33)))
    return img2, img22, img3, img33


# 均值滤波器
def ask11(img):
    h, w, c = img.shape
    img1 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    # print(img1.shape)
    img2 = numpy.zeros_like(img)

    for x in range(h):
        for y in range(w):
            img2[x, y] = numpy.mean(img1[x:x + 3, y:y + 3], axis=(0, 1))

    cv_show("compare", np.hstack((img, img2)))


# Motion Filter
def ask12(img):
    h, w, c = img.shape
    img1 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT).astype(int)
    img2 = np.zeros_like(img)

    for x in range(h):
        for y in range(w):
            img2[x, y] = (img1[x, y] + img1[x + 1, y + 1] + img1[x + 2, y + 2]) / 3
            # print("1",img1[x, y] + img1[x + 1, y + 1] + img1[x + 2, y + 2])
            # print(img1[x:x+3,y:y+3])
            # print(img2[x,y])
            # print()

    cv_show("compare", np.hstack((img, img2)))


# MAX-MIN滤波器
def ask13(img):
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    img2 = np.zeros_like(gray)

    for x in range(h):
        for y in range(w):
            img2[x, y] = np.max(img1[x:x + 3, y:y + 3]) - np.min(img1[x:x + 3, y:y + 3])

    cv_show("compare", np.hstack((gray, img2)))


# 差分滤波器
def ask14(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.zeros([3, 3])
    k[0, 1] = -1
    k[1, 1] = 1

    convolutin(k, gray)

    img4 = cv2.Laplacian(gray, cv2.CV_8U, ksize=3, borderType=cv2.BORDER_REFLECT)
    cv_show("img4", img4)


# Sobel滤波器
def ask15(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convolutin(k, gray)


# Prewitt滤波器
def ask16(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    convolutin(k, gray)


# Laplacian滤波器
def ask17(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img2, img22, _, _ = convolutin(k, gray)

    img1=cv2.Laplacian(gray,-1,ksize=1)
    cv_show("compare",np.hstack((img1,img2,img22)))

#Emboss滤波器
def ask18(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    img2, img22, _, _ = convolutin(k, gray)

    img1 = cv2.Laplacian(gray, -1, ksize=1)
    cv_show("compare", np.hstack((img1, img2, img22)))

#LoG滤波器
def ask19(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2=cv2.GaussianBlur(img1,ksize=(5,5),sigmaX=3,sigmaY=3)
    cv_show("img2",img2)

    img2=cv2.Laplacian(img2,-1,ksize=3)
    cv_show("img2",img2)

    img3=np.clip(img1+img2,0,255)
    cv_show("img3",img3)
    print(img.var())

    print("-----下面是官方-----")
    K_size,sigma=5,3
    H, W, C = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img1.copy().astype(np.float)
    tmp = out.copy()

    # LoG Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()

    # filtering
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    cv_show("out",out)

def graysharpness(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv_show("img_gray", img_gray)
    # img1 = cv2.Laplacian(img_gray, -1, ksize=3)
    # cv_show("img1", img1)

    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    img2, img22, _, _ = convolutin(k, img_gray)

    img3=np.clip(img_gray.astype(float) + img22.astype(float),0,255).astype("uint8")
    cv_show("compare", np.hstack((img_gray,img3)))

    # img2 = np.clip(img_gray + img22, 0, 255)
    # cv_show("img2", img2)

def ask20(img):
    import matplotlib.pyplot as plt
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])

    plt.plot(range(256), hist)
    y_maxValue = np.max(hist)
    plt.axis([0, 255, -0, y_maxValue])
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()

    print("-----下面是官方-----")
    #plt.hist(img.ravel(),bins=255,rwidth=0.8,range=(0,255))
    plt.hist(img_gray.ravel(),bins=255,rwidth=0.8,range=(0,255))
    plt.show()

if __name__ == "__main__":
    img = cv2.imread("imori.jpg")
    img1 = cv2.imread("../lena_color_512.tif")
    img_noise=cv2.imread("imori_noise.jpg")
    #ask18(img)
    graysharpness(img1)
    #ask19(img_noise)
    # ask17(img)