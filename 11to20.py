import cv2
import numpy
import numpy as np

def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.waitKey(0)

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

    cv_show("compare",np.hstack((grayimg,img2,img22,img3,img33)))


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

    convolutin(k,gray)

    img4=cv2.Laplacian(gray,cv2.CV_8U,ksize=3,borderType=cv2.BORDER_REFLECT)
    cv_show("img4",img4)

#Sobel滤波器
def ask15(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    convolutin(k,gray)



if __name__ == "__main__":
    img = cv2.imread("imori.jpg")
    ask14(img)
