import cv2
import numpy as np
import random
from astar import astar
from dij import dij
import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# 加载图像
img = cv2.imread('./map.png', cv2.IMREAD_COLOR)
img1 = cv2.imread('./map1.png', cv2.IMREAD_COLOR)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找到障碍物
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# 对障碍物进行膨胀处理，确保路径不会穿过它们
kernel = np.ones((1, 1), np.uint8)
inflated_obstacles = cv2.dilate(binary, kernel, iterations=2)

# 找到代表障碍物的轮廓
contours, _ = cv2.findContours(inflated_obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在网格上表示障碍物
grid = np.zeros_like(gray)
for contour in contours:
    cv2.drawContours(grid, [contour], 0, (255), -1)

output_path = './gray_map.png'
cv2.imwrite(output_path, grid)
# 在图像上标记障碍物的边缘为黄色
for contour in contours:
    cv2.drawContours(img, [contour], -1, (0, 255, 255), 1)

# 在网格中找到合适的起点和终点
height, width = grid.shape

# 在网格中找到一个空白区域作为起点或终点
def find_free_space(grid):
    while True:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if grid[y, x] == 0:
            return (x, y)


start = find_free_space(grid)
end = find_free_space(grid)

# start = (0, 0)
# end = (5, 50)

cv2.circle(img, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
cv2.circle(img, end, 2, (0, 0, 255), -1)    # 终点用红色表示
cv2.circle(img1, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
cv2.circle(img1, end, 2, (0, 0, 255), -1)    # 终点用红色表示
output_path_corrected = './map_with_path_and_obstacles0.png'
cv2.imwrite(output_path_corrected, img)

# 将网格定义为0表示空白空间,255表示障碍物
grid = np.where(grid == 255, 1, 0)

path_a, ee = astar(grid, start, end)
path_b, ee = dij(grid, start, end)

# 绘图
if path_a:
    for i in range(1, len(path_a)):
        cv2.line(img, path_a[i - 1], path_a[i], (0, 255, 0), 2)  # 绘制绿色线条表示路径

    # 绘制起点和终点
    cv2.circle(img, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
    cv2.circle(img, end, 2, (0, 0, 255), -1)    # 终点用红色表示

# 保存结果
output_path_corrected = './astar_map_with_path_and_obstacles1.png'
cv2.imwrite(output_path_corrected, img)

if path_b:
    for i in range(1, len(path_b)):
        cv2.line(img1, path_b[i - 1], path_b[i], (255, 255, 0), 2)  # 绘制绿色线条表示路径

    # 绘制起点和终点
    cv2.circle(img1, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
    cv2.circle(img1, end, 2, (0, 0, 255), -1)    # 终点用红色表示

# 保存结果
output_path_corrected = './dij_map_with_path_and_obstacles1.png'
cv2.imwrite(output_path_corrected, img1)
