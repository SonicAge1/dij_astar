import cv2
import numpy as np
import heapq
import random


# 启发式函数用于估算在网格中从当前点到终点的距离
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# 节点类,用于存储父节点,自身位置,G/H/F值
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


# dij算法实现
def dij(maze, start, end):
    # 初始化起始和结束节点
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    explored_nodes = 0  # 添加一个计数器

    # 初始化open和closed列表
    open_list = []
    closed_set = set()

    # 将起始节点添加到open列表
    heapq.heappush(open_list, (start_node.f, start_node))

    # 循环直到找到终点
    while open_list:
        # 获取当前节点(open列表中f值最小的节点)
        current_node = heapq.heappop(open_list)[1]
        explored_nodes += 1  # 增加计数器

        # 找到终点,重建路径
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], explored_nodes  # 返回路径

        # 生成子节点
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # 确保子节点在网格范围内
            if node_position[0] >= len(maze[0]) or node_position[0] < 0 or node_position[1] >= len(maze) or node_position[1] < 0:
                continue

            # 确保子节点不是障碍物
            if maze[node_position[1]][node_position[0]] != 0:
                continue

            # 创建新节点
            new_node = Node(current_node, node_position)

            # 如果子节点在closed列表,跳过
            if new_node.position in closed_set:
                continue

            # 子节点的g,h,f值
            if new_position in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                # 对角线移动的成本
                new_node.g = current_node.g + np.sqrt(2)
            else:
                # 直线移动的成本
                new_node.g = current_node.g + 1

            new_node.h = heuristic(end_node.position, new_node.position)
            new_node.f = new_node.g # + new_node.h

            # 检查子节点是否已在open列表中且具有更高的g值
            existing_node = None

            # 遍历开放列表中的每个节点
            for _, child in open_list:
                # 检查节点的位置是否与新节点的位置相同
                if child.position == new_node.position:
                    # 如果找到了,更新existing_node并退出循环
                    existing_node = child
                    break

            # 如果找到了现有的节点并且它的G值小于或等于新节点的G值
            if existing_node is not None and existing_node.g <= new_node.g:
                # 那么新节点不提供更好的路径,忽略这个新节点
                pass
            else:
                # 否则,将新节点添加到开放列表中
                heapq.heappush(open_list, (new_node.f, new_node))

        # 将当前节点移到closed列表
        closed_set.add(current_node.position)

    # 如果没有找到路径
    return None, explored_nodes

# # 加载图像
# img = cv2.imread('./map.png', cv2.IMREAD_COLOR)
#
# # 转换为灰度图
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 在图像中找到障碍物
# _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#
# # 对障碍物进行膨胀处理，确保路径不会穿过它们
# kernel = np.ones((1, 1), np.uint8)
# inflated_obstacles = cv2.dilate(binary, kernel, iterations=2)
#
# # 找到代表障碍物的轮廓
# contours, _ = cv2.findContours(inflated_obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 在网格上表示障碍物
# grid = np.zeros_like(gray)
# for contour in contours:
#     cv2.drawContours(grid, [contour], 0, (255), -1)
#
# output_path = './gray_map.png'
# cv2.imwrite(output_path, grid)
# # 在图像上标记障碍物的边缘为黄色
# for contour in contours:
#     cv2.drawContours(img, [contour], -1, (0, 255, 255), 1)
#
# # 在网格中找到合适的起点和终点
# height, width = grid.shape
#
# # 在网格中找到一个空白区域作为起点或终点
# def find_free_space(grid):
#     while True:
#         x = random.randint(0, width - 1)
#         y = random.randint(0, height - 1)
#         if grid[y, x] == 0:
#             return (x, y)
#
#
# start = find_free_space(grid)
# end = find_free_space(grid)
#
# # start = (0, 0)
# # end = (5, 50)
#
# cv2.circle(img, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
# cv2.circle(img, end, 2, (0, 0, 255), -1)    # 终点用红色表示
# output_path_corrected = './dij_map_with_path_and_obstacles0.png'
# cv2.imwrite(output_path_corrected, img)
#
# # 将网格定义为0表示空白空间,255表示障碍物
# grid = np.where(grid == 255, 1, 0)
#
# # 运行A*算法
# path = dij(grid, start, end)
#
# # 如果找到路径,则在图像上绘制
# if path:
#     for i in range(1, len(path)):
#         cv2.line(img, path[i - 1], path[i], (0, 255, 0), 2)  # 绘制绿色线条表示路径
#
#     # 绘制起点和终点
#     cv2.circle(img, start, 2, (255, 0, 0), -1)  # 起点用蓝色表示
#     cv2.circle(img, end, 2, (0, 0, 255), -1)    # 终点用红色表示
#
# # 保存结果
# output_path_corrected = './dij_map_with_path_and_obstacles1.png'
# cv2.imwrite(output_path_corrected, img)
