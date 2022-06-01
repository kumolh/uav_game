from turtle import color
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def view_reward():
    img = Image.open(r'./assets/view.png')
    img = img.filter(ImageFilter.BoxBlur(4))
    h, w = img.size
    draw = ImageDraw.Draw(img)
    draw.line((0, h, w/2, h/2), fill=(255, 255, 255))
    draw.line((w/2, h/2, w, h), fill=(255, 255, 255))
    img.show()

def location_reward():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = y1 = np.arange(-100, 100, 1) #np.hstack([np.arange(-100, -32, 1), np.arange(32, 100, 1)])
    X, Y = np.meshgrid(x1, y1)
    z1 = 100 * np.exp(-.0006 * (X**2 + Y**2))
    pivot = 100 * np.exp(-.0006 * 32**2)
    for i in range(len(z1)):
        for j in range(len(z1[0])):
            if X[i][j] ** 2 + Y[i][j] ** 2 <= 32 ** 2:
                z1[i][j] = pivot
    surf = ax.plot_surface(X, Y, z1, cmap=cm.coolwarm)
    # Customize the z axis.
    ax.set_zlim(-5, 55)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)    
    # img = mpimg.imread()
    # img = Image.open('./assets/bottom.png')
    # pix = img.load()
    # w, h = img.size
    # x = np.arange(0, w)
    # y = np.arange(0, h)
    # x, y = np.meshgrid(x, y)
    # z = np.zeros(w * h)
    # colors = []
    # for i in range(w):
    #     for j in range(h):
    #         rgb = tuple(np.array(pix[i, j])/ 255)
    #         colors.append(rgb)
    # ax.scatter(x,y,z, c= colors)
    plt.show()

if __name__ == '__main__':
    location_reward()