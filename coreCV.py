import cv2 as cv
import numpy as np
import pandas as pd

border_wight = 3
line_width = 15
filter_width = 5
line_corner_interval = 3
width = 5000
height = 5000
# 判断直线像素阈值
same_row_threshold = 30
# 网格横轴cell数量
grid_row_cell_count = 24
# 网格纵轴cell数量
grid_col_cell_count = 18
# 横轴点数
grid_row_point_count = grid_row_cell_count * 2 + 1
# 纵轴点数
grid_col_point_count = grid_col_cell_count * 2 + 1
# 每一个格子实际长度，默认5mm
cell_real_length = 5
# 分辨率
pic_pix = 600
# 英寸长度
one_yc = 2.54

one_cell_pix = pic_pix * (cell_real_length / 10) / one_yc


def pre_handle(img) -> any:
    # 高斯模糊
    img_gas = cv.GaussianBlur(img, (13, 13), 0)
    # 边缘检测
    img_edges = cv.Canny(img_gas, 200, 255)
    return img_edges


# Press the green button in the gutter to run the script.


def convert_pix_to_mm(pix):
    return round(10 * one_yc * pix / pic_pix, 2)


def exist_neighbor(source, target):
    td = 30
    _, _, _, _, _, sx, sy = source
    _, _, _, _, _, tx, ty = target
    if sx == tx and sy == ty:
        return False
    if source[2] <= filter_width or source[3] <= filter_width:
        return False
    return abs(sx - tx) < td and abs(sy - ty) < td
    # return pow(abs(sx - tx), 2) + pow(abs(sy - ty), 2) <= pow(td, 2)


def remove_np_array(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def merge_area(neighbors, target) -> any:
    neighbors.append(target)
    min_x = min(neighbors, key=lambda xx: xx[0])[0]
    min_y = min(neighbors, key=lambda xx: xx[1])[1]
    max_xt = max(neighbors, key=lambda xx: xx[0] + xx[2])
    max_x = max_xt[0] + max_xt[2]
    max_yt = max(neighbors, key=lambda xx: xx[1] + xx[3])
    max_y = max_yt[1] + max_yt[3]
    w = max_x - min_x
    h = max_y - min_y
    s = w * h
    mid_x = int((min_x + max_x) / 2)
    mid_y = int((min_y + max_y) / 2)
    new = np.array(
        [min_x, min_y, w, h, s, mid_x, mid_y])
    return new


def fix_border_point(merge) -> any:
    x, y, w, h, s, mid_x, mid_y = merge
    if w > h:
        if y < height / 2:
            # 处理上边界
            y = y + h - w
        h = w
    else:
        if x < width / 2:
            x = x + w - h
        w = h
    s = w * h
    mid_x = int(x + w / 2)
    mid_y = int(y + h / 2)
    return [x, y, w, h, s, mid_x, mid_y]


def handle_stats(stats, centroids):
    res = []
    stats_mid = []
    for a, c in zip(stats, centroids):
        fx, fy = c
        # 添加中心点坐标
        a = np.append(a, [int(fx), int(fy)])
        # if a[2] <= filter_width or a[3] <= filter_width:
        #     continue
        stats_mid.append(a.tolist())
    for a in stats_mid:
        # 过滤掉已有连通区域中心阈值半径以内的连通区域
        neighbors = list(filter(lambda x: exist_neighbor(x, a), stats_mid))
        if len(neighbors) > 0:
            merge = merge_area(neighbors, a)
            # 处理边上的交点,还原成一个正方形
            if abs(merge[2] - merge[3]) > line_corner_interval:
                merge = fix_border_point(merge)
            res.append(merge)
        else:
            # 处理边上的交点,还原成一个正方形
            if abs(a[2] - a[3]) > line_corner_interval:
                a = fix_border_point(a)
            if a[2] > line_width and a[3] > line_width:
                merge = a
                res.append(merge)
    res = list(set([tuple(t) for t in res]))
    res = [list(v) for v in res]
    return res


def print_err_line_point(img_rgb, cur_row):
    img_err = cv.imread(handle_path)
    for error_p in cur_row:
        cv.circle(img_err, (error_p[0], error_p[1]), 10, (255, 0, 0), -1)
        cv.circle(img_rgb, (error_p[0], error_p[1]), 10, (255, 0, 0), -1)
    cv.imwrite('err.png', img_err)
    cv.imwrite('err1.png', img_rgb)
    pass


def handle_x_row(y_index, y_point, cur_row):
    if len(cur_row) < grid_row_point_count:
        print_err_line_point(cur_row)
        raise 'y=' + str(y_index) + '识别的点位不足'
    x_row_right = list(filter(lambda f: f[0] > y_point[0], cur_row))
    x_row_right = sorted(x_row_right, key=lambda xx: xx[0])
    right_index = 0
    move_point_x = y_point[0] + one_cell_pix
    while right_index < grid_row_cell_count:
        if len(x_row_right) == 0:
            break
        near = min(x_row_right, key=lambda f: abs(f[0] - move_point_x))
        interval = abs(near[0] - move_point_x)
        if abs(interval) < 30:
            move_point_x = near[0] + one_cell_pix
            right_index = right_index + 1
            list.append(near, right_index)
            list.append(near, y_index)
            x_row_right = list(filter(lambda f: f[0] > near[0], x_row_right))
        else:
            raise '误差太大'
    if right_index < grid_row_cell_count:
        raise '处理有效点位少于' + str(grid_row_cell_count)
    x_row_left = list(filter(lambda f: f[0] < y_point[0], cur_row))
    x_row_left = sorted(x_row_left, key=lambda xx: xx[0], reverse=True)

    left_index = 0
    move_point_x = y_point[0] - one_cell_pix
    while left_index > -1 * grid_row_cell_count:
        if len(x_row_left) == 0:
            break
        near = min(x_row_left, key=lambda f: abs(f[0] - move_point_x))
        interval = abs(near[0] - move_point_x)
        if abs(interval) < 30:
            move_point_x = near[0] - one_cell_pix
            left_index = left_index - 1
            list.append(near, left_index)
            list.append(near, y_index)
            x_row_left = list(filter(lambda f: f[0] < near[0], x_row_left))
        else:
            raise '误差太大'
    if left_index > -1 * grid_row_cell_count:
        raise '处理有效点位少于' + str(grid_row_cell_count)
    skip = list(filter(lambda f: len(f) < 4, cur_row))
    for sk in skip:
        cur_row.remove(sk)


def fix_original_point(coordinate: list):
    min_point = min(coordinate, key=lambda xx: xx[0] + xx[1])
    max_point = max(coordinate, key=lambda xx: xx[0] + xx[1])
    centre = [(min_point[0] + max_point[0]) / 2, (min_point[1] + max_point[1]) / 2]
    origin_point = min(coordinate, key=lambda c: pow(c[0] - centre[0], 2) + pow(c[1] - centre[1], 2))
    print("原点校正前" + str(origin_point))
    # 原点校正
    x_row = list(filter(lambda f: abs(f[1] - origin_point[1]) < same_row_threshold, coordinate))
    x_row.sort(key=lambda f: f[0])
    if len(x_row) != grid_row_point_count:
        raise 'x轴校正失败'
    if origin_point[0] != x_row[grid_row_cell_count][0]:
        origin_point = x_row[grid_row_cell_count]
        print('x轴已校正')
    y_row = list(filter(lambda f: abs(f[0] - origin_point[0]) < same_row_threshold, coordinate))
    y_row.sort(key=lambda f: f[1])
    if len(y_row) != grid_col_point_count:
        print_err_line_point(y_row)
        raise 'y轴校正失败'
    if origin_point[1] != y_row[grid_col_cell_count][1]:
        origin_point = y_row[grid_col_cell_count]
        print('y轴已校正')
    print("原点校正后" + str(origin_point))
    return origin_point


def create_coordinate(coordinate: list):
    origin_point = fix_original_point(coordinate)
    print("原点校正前" + str(origin_point))
    # 原点校正
    x_row = list(filter(lambda f: abs(f[1] - origin_point[1]) < same_row_threshold, coordinate))
    x_row.sort(key=lambda f: f[0])
    y_row = list(filter(lambda f: abs(f[0] - origin_point[0]) < same_row_threshold, coordinate))
    y_row.sort(key=lambda f: f[1])
    list.append(origin_point, 0)
    list.append(origin_point, 0)
    # 找到原点所属的横坐标上所有交点
    # x_row = list(filter(lambda f: abs(f[1] - origin_point[1]) < same_row_threshold, coordinate))
    # 处理x轴
    handle_x_row(0, origin_point, x_row)
    x_row.sort(key=lambda f: f[0])
    # 沿y轴正向
    y_row = list(filter(lambda f: abs(f[0] - origin_point[0]) < same_row_threshold, coordinate))
    y_row_right = list(filter(lambda f: f[1] > origin_point[1], y_row))
    y_row_right = sorted(y_row_right, key=lambda xx: xx[1])
    y_right_index = 0
    for yr in y_row_right:
        y_right_index = y_right_index + 1
        list.append(yr, 0)
        list.append(yr, y_right_index)
        # 处理当前y=p所在的x轴所有点
        cur_row = list(filter(lambda f: abs(f[1] - yr[1]) < same_row_threshold, coordinate))
        handle_x_row(y_right_index, yr, cur_row)
        cur_row.sort(key=lambda f: f[0])
    # 沿y轴负向
    y_row_left = list(filter(lambda f: f[1] < origin_point[1], y_row))
    y_row_left = sorted(y_row_left, key=lambda xx: xx[1], reverse=True)
    y_left_index = 0
    for yl in y_row_left:
        y_left_index = y_left_index - 1
        list.append(yl, 0)
        list.append(yl, y_left_index)
        # 处理当前y=p所在的x轴所有点
        cur_row = list(filter(lambda f: abs(f[1] - yl[1]) < same_row_threshold, coordinate))
        handle_x_row(y_left_index, yl, cur_row)
        cur_row.sort(key=lambda f: f[0])
    skip = list(filter(lambda f: len(f) < 4, coordinate))
    for sk in skip:
        coordinate.remove(sk)
    return coordinate, origin_point, skip


def save_to_excel(data_list):
    # 二维list
    # list转dataframe
    df = pd.DataFrame(data_list, columns=['交点像素X坐标',
                                          '交点像素Y坐标',
                                          'X轴坐标',
                                          'Y轴坐标',
                                          'X维度模版距离(mm)',
                                          'Y维度模版距离(mm)',
                                          'X方向像素距离',
                                          'Y方向像素距离',
                                          'X方向像素长度',
                                          'Y方向像素长度',
                                          'X方向偏离值(mm)',
                                          'Y方向偏离值(mm)'])
    # 保存到本地excel
    df.to_excel("网格标定偏离值计算.xlsx", index=False)


def save_offset_to_excel(data_list):
    if len(data_list) <= 0:
        raise "结果为空"
    fd = data_list[0]
    cols = len(fd)
    col_names = ['行/列']
    for col_idx in range(1, cols):
        zh = col_idx // 2
        if col_idx % 2 == 1:
            col_names.append('列' + str(zh + 1) + '_X')
        else:
            col_names.append('列' + str(zh) + '_Y')

    df = pd.DataFrame(data_list, columns=col_names)
    # 保存到本地excel
    df.to_excel("偏离值汇总.xlsx", index=False)
    pass


def write_png(img_rgb, img_original, merge_stats, original_point, skip, mode):
    valid_count = 0
    for a in merge_stats:
        if mode == 'original':
            x, y, w, h, s = a
        else:
            x, y, w, h, s, mid_x, mid_y = a
            if len(skip) > 0 and len(list(filter(lambda f: f[0] == mid_x and f[1] == mid_y, skip))) > 0:
                continue
        # x, y, w, h, s = a
        if s > 10000:
            continue
        valid_count = valid_count + 1
        # 中心点坐标
        cv.circle(img_original, (x, y), 2, (255, 0, 0), -1)
        cv.circle(img_original, (x + w, y), 2, (255, 0, 0), -1)
        cv.circle(img_original, (x, y + h), 2, (255, 0, 0), -1)
        cv.circle(img_original, (x + w, y + h), 2, (255, 0, 0), -1)
        cv.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.circle(img_original, (x + int(w / 2), y + int(h / 2)), 5, (0, 0, 255), -1)

        cv.circle(img_rgb, (x, y), border_wight, (255, 0, 0), -1)
        cv.circle(img_rgb, (x + w, y), border_wight, (255, 0, 0), -1)
        cv.circle(img_rgb, (x, y + h), border_wight, (255, 0, 0), -1)
        cv.circle(img_rgb, (x + w, y + h), border_wight, (255, 0, 0), -1)
        cv.circle(img_rgb, (original_point[0], original_point[1]), 10, (0, 255, 0), -1)
        cv.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imwrite('original-after-bd.png', img_original)
    cv.imwrite('original-after-bd-red.png', img_rgb)
    print('valid count ' + str(valid_count))


def get_original(coordinate):
    min_point = min(coordinate, key=lambda xx: xx[0] + xx[1])
    max_point = max(coordinate, key=lambda xx: xx[0] + xx[1])
    centre = [(min_point[0] + max_point[0]) / 2, (min_point[1] + max_point[1]) / 2]
    origin_point = min(coordinate, key=lambda c: pow(c[0] - centre[0], 2) + pow(c[1] - centre[1], 2))
    return origin_point


# img_original = None
# img_rgb = None
handle_path = None


def handle_entry(handle_path):
    img_rgb = cv.imread(handle_path)
    img_rgb[:, :, 2] = 0
    img = pre_handle(img_rgb)
    gray = np.float32(img)
    dst = cv.cornerHarris(gray, 13, 3, 0.05)
    hls_max = dst.max()
    # 3 设置阈值，将角点绘制出来，阈值根据图像进行选择
    img_rgb[dst > 0.16 * hls_max] = [0, 0, 255]
    # 已经在rgb图像上勾勒出交叉区域
    hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red = np.array([0, 255, 255])
    upper_red = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower_red, upper_red)
    res = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
    cv.imwrite('red.png', res)
    # res = cv.imread('red.png')
    res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(res_gray, 50, 255, 0)
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    bin_clo = cv.dilate(binary, kernel2, iterations=2)
    # 获取所有的连通域
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(bin_clo, connectivity=8)
    # 连通域合并
    merge_stats = handle_stats(stats, centroids)
    # 画出所有连通域的外接矩形
    # 重新创建原图
    img_original = cv.imread(handle_path)
    coordinates = []
    for a in merge_stats:
        x, y, w, h, s, mid_x, mid_y = a
        # x, y, w, h, s = a
        if s > 10000:
            continue
        # 中心点坐标
        cur_coordinate = [mid_x, mid_y]
        coordinates.append(cur_coordinate)
    op = fix_original_point(coordinates)
    # write_png(merge_stats, op, [], 'handled')
    # 建立坐标系
    crd, original_point, skip = create_coordinate(coordinates)
    # 打印图像标定
    write_png(img_rgb, img_original, merge_stats, original_point, skip, 'handled')
    # 计算距离
    crd.sort(key=lambda u: (u[3], u[2]))
    for p in crd:
        pix_x = p[0] - original_point[0]
        pix_y = p[1] - original_point[1]
        list.append(p, cell_real_length * p[2])
        list.append(p, cell_real_length * p[3])
        list.append(p, pix_x)
        list.append(p, pix_y)
        list.append(p, convert_pix_to_mm(pix_x))
        list.append(p, convert_pix_to_mm(pix_y))
        list.append(p, round(p[4] - p[8], 3))
        list.append(p, round(p[5] - p[9], 3))
        # print(p)
    # 输出原始数据
    save_to_excel(crd)
    # 获取所有纵坐标
    y_set = list(map(lambda f: f[3], crd))
    y_set = list(set(y_set))
    y_set.sort()
    row_index = 0
    offset = []
    for y_value in y_set:
        row_index = row_index + 1
        cur_row = []
        row = list(filter(lambda f: f[3] == y_value, crd))
        row.sort(key=lambda f: f[2])
        cur_row.append('行' + str(row_index))
        for point in row:
            cur_row.append(point[10])
            cur_row.append(point[11])
        offset.append(cur_row)
    # 输出偏离值结果数据
    save_offset_to_excel(offset)

def trans_to_red(path):
    img = cv.imread("img.jpg")
