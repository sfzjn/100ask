import cv2

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)

#通道交换
def ask1(img):
    b,g,r=img[:,:,0].copy(),img[:,:,1].copy(),img[:,:,2].copy()
    img[:, :, 0], img[:, :, 1], img[:, :, 2]=r,g,b
    cv_show("ask1",img)

    img=img[:,:,(2,1,0)]
    cv_show("ask11",img)

#灰度化
def ask2(img):
    b,g,r=img[:,:,0].copy(),img[:,:,1].copy(),img[:,:,2].copy()


if __name__=="__main__":
    img=cv2.imread("imori.jpg")
    ask1(img)