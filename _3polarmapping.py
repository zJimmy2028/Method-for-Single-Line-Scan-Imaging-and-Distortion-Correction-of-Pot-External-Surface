import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cartesian_to_polar(x, y, width):
    theta = 2 * np.pi * x / width
    r = y
    return r, theta

def transform(image):
    height, width = image.shape
    radius = height
    d = 2 * radius

    x = np.arange(width)
    y = np.arange(height)
    xp, yp = np.meshgrid(x, y)  # 创建网格坐标矩阵

    # polar_img = np.zeros((d, d), dtype=np.uint8)

    r, theta = cartesian_to_polar(xp, yp, width)  # 转换为极坐标
    new_x, new_y = polar_to_cartesian(r, theta)  # 转换为笛卡尔坐标
    new_x = np.round(new_x + radius).astype(int)  # 四舍五入并转换为整数
    new_y = np.round(new_y + radius).astype(int)

    valid_indices = (new_x >= 0) & (new_x < d) & (new_y >= 0) & (new_y < d)
    
    # 原图像的有效坐标和像素值
    valid_xp = xp[valid_indices]
    valid_yp = yp[valid_indices]
    valid_pixel_values = image[valid_yp, valid_xp]

    # 创建极坐标图像坐标网格
    grid_x, grid_y = np.meshgrid(np.arange(d), np.arange(d))

    # 使用插值填补极坐标图像
    polar_img = griddata(
        (new_x[valid_indices], new_y[valid_indices]),
        valid_pixel_values,
        (grid_x, grid_y),
        method='linear'  # 选择插值方法
        # method = 'nearest'
    )
    # 处理 NaN 值，填充为零
    polar_img = np.nan_to_num(polar_img, nan=0)
    return polar_img

if __name__ == '__main__':
    img = cv2.imread('image/123.jpg', cv2.IMREAD_GRAYSCALE)
    tra_img = transform(img)
    print('img:', img.shape, '\t tra_img:', tra_img.shape)
    cv2.imwrite('image/123_polar.jpg', tra_img)
    plt.imshow(tra_img, cmap='gray')
    plt.show()  
