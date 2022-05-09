import numpy as np
import cv2
import common
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

color_dict = {0: "Gray", 1: "Blue", 2: "Green", 3: "Red"}

def calcPRNU(img_color):
    def PRNU(tmplist):
        avg = np.average(tmplist)
        std = np.std(tmplist)
        return round(std / avg,5)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    label = ["Gray", "Red", "Green", "Blue"]
    ravel = [img_gray.ravel(), img_color[:, :, 2].ravel(), img_color[:, :, 1].ravel(), img_color[:, :, 0].ravel()]

    for i in range(len(label)):
        print("{} PRNU:{}".format(label[i], PRNU(ravel[i])))


def plot3D(file_path):
    img_color = cv2.imread(file_path)
    print(type(img_color))

    if str(type(img_color)) != "<class 'numpy.ndarray'>":
        print("图像打开失败")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    tm = common.TimeCost()
    ax = np.arange(0, h).reshape(h, 1)
    ax = np.repeat(ax, w, axis=1)
    ay = np.arange(0, w).reshape(1, w)
    ay = np.repeat(ay, h, axis=0)
    print(tm.getCostTime())

    # tm=TimeCost()
    # ax, ay = [], []
    # for x in range(h):
    #     for y in range(w):
    #         ax.append(x)
    #         ay.append(y)
    # ax = np.array(ax).reshape([h, w])
    # ay = np.array(ay).reshape([h, w])
    #
    # print(tm.getCostTime())
    az = []
    tm1 = common.TimeCost()
    style.use('ggplot')
    fig = plt.figure(dpi=100)
    figure = []
    for i in range(4):
        figure.append(fig.add_subplot(2, 2, i + 1, projection='3d'))
        figure[i].set_xlabel('x axis')
        figure[i].set_ylabel('y axis')
        figure[i].set_zlabel('z axis')
        if i == 0:
            az.append(img_gray[:, :])
        else:
            az.append(img_color[:, :, i - 1])
        figure[i].plot_wireframe(ax, ay, az[i], rstride=50, cstride=100)
        plt.title(color_dict[i])

        print("{}平均:".format(color_dict[i]), np.mean(az[i]))
        print("{}方差：".format(color_dict[i]), np.var(az[i]))
        max_index = np.argmax(az[i])
        max_x = max_index // w
        max_y = max_index - w * max_x
        print("{}最大值：{} 坐标：({},{})".format(color_dict[i], np.max(az[i]), max_x, max_y))
        print("{}最小值：".format(color_dict[i]), np.min(az[i]))
        print("{}标准差：".format(color_dict[i]), np.std(az[i]))
        print("-------------------")

    print(tm1.getCostTime())

    # x, y, z = axes3d.get_test_data()
    # print(axes3d.__file__)
    # .plot_surface(ax, ay, az, rstride=50, cstride=50)

    plt.show()

def calcHist(img_color):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    plt.hist(img_gray.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    # plt.hist(img_color[:,:,2].ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.show()

def check_nonuniformity(file_path):
    img=cv2.imread(file_path)
    print(type(img))

    if str(type(img))=="<class 'numpy.ndarray'>":
        calcPRNU(img)
        # plot3D(img)
    else:
        print("图像打开失败")

def plot3D1():
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x, y, z = axes3d.get_test_data()

    print(axes3d.__file__)
    ax1.plot_wireframe(x, y, z, rstride=3, cstride=3)

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')

    plt.show()