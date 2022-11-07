import math

import numpy as np


class Point(object):
    x = 0
    y = 0
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    # 判断点坐标是否为0
    def is_empty(self):
        if self.x == 0 and self.y == 0:
            return False
        else:
            return True




# 三个点求角度
def vector_angle(p1, p, p2):
    # get vector
    # v1 = np.array([(p1.x - p.x), (p1.y - p.y)])
    # v2 = np.array([(p2.x - p.x), (p2.y - p.y)])
    # cost = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # angle = 180 - np.arccos(cost)

    dx1 = p1.x - p.x
    dy1 = p1.y - p.y
    dx2 = p2.x - p.x
    dy2 = p2.y - p.y
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


# 两个向量求角度
def two_vector_angle(v1, v2):
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = math.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * math.sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    angle = math.acos(cos)/ math.pi * 180
    return angle


def get_angles(keypoints):

    angles = []

    # get points
    left_shoulder = Point(keypoints[2][0], keypoints[2][1])
    right_shoulder = Point(keypoints[5][0], keypoints[5][1])
    left_elbow = Point(keypoints[3][0], keypoints[3][1])
    right_elbow = Point(keypoints[6][0], keypoints[6][1])
    left_wrist = Point(keypoints[4][0], keypoints[4][1])
    right_wrist = Point(keypoints[7][0], keypoints[7][1])
    left_hip = Point(keypoints[8][0], keypoints[8][1])
    right_hip = Point(keypoints[11][0], keypoints[11][1])
    left_knee = Point(keypoints[9][0], keypoints[9][1])
    right_knee = Point(keypoints[12][0], keypoints[12][1])
    left_foot = Point(keypoints[10][0], keypoints[10][1])
    right_foot = Point(keypoints[13][0], keypoints[13][1])

    # get angles
    angles.append(vector_angle(left_shoulder, left_elbow, left_wrist)) # 1
    angles.append(vector_angle(right_shoulder, right_elbow, right_wrist)) # 2
    angles.append(vector_angle(left_elbow, left_shoulder, left_hip)) # 3
    angles.append(vector_angle(right_elbow, right_shoulder, right_hip)) # 4
    angles.append(vector_angle(left_shoulder, left_hip, left_knee)) # 5
    angles.append(vector_angle(right_shoulder, right_hip, right_knee)) # 6
    angles.append(vector_angle(left_hip, left_knee, left_foot)) # 7
    angles.append(vector_angle(right_hip, right_knee, right_foot)) # 8

    return angles


# 判断两条线段是否相交
def cross(p1,p2,p3):#跨立实验
    x1=p2.x-p1.x
    y1=p2.y-p1.y
    x2=p3.x-p1.x
    y2=p3.y-p1.y
    return x1*y2-x2*y1


def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1.x,p2.x)>=min(p3.x,p4.x)    #矩形1最右端大于矩形2最左端
    and max(p3.x,p4.x)>=min(p1.x,p2.x)   #矩形2最右端大于矩形最左端
    and max(p1.y,p2.y)>=min(p3.y,p4.y)   #矩形1最高端大于矩形最低端
    and max(p3.y,p4.y)>=min(p1.y,p2.y)): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
    else:
        D=0
    return D


# 两点求欧氏距离
def getDist_P2P(p1,p2):
    distance=math.pow((p1.x - p2.x),2) + math.pow((p1.y - p2.y),2)
    distance=math.sqrt(distance)
    return distance


# 求点到线的距离
def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2.y - line_point1.y
    B = line_point1.x - line_point2.x
    C = (line_point1.y - line_point2.y) * line_point1.x + \
        (line_point2.x - line_point1.x) * line_point1.y
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point.x + B * point.y + C) / (np.sqrt(A**2 + B**2))
    return distance


def judge_function(keypoints):
    labels = ['None', 'None', 'None', 'None']

    # get angles
    keypoints = keypoints[0]
    angles = get_angles(keypoints)
    print('angles: ', angles)

    # center_point = Point(keypoints[1][0], keypoints[1][1])
    nose = Point(keypoints[0][0], keypoints[0][1])
    left_shoulder = Point(keypoints[2][0], keypoints[2][1])
    right_shoulder = Point(keypoints[5][0], keypoints[5][1])
    lrs_Dist = getDist_P2P(left_shoulder, right_shoulder)
    p2l_Dist = get_distance_from_point_to_line(nose, left_shoulder, right_shoulder)
    print('lrs_dist:', lrs_Dist)
    print('p2l_dist', p2l_Dist)

    # 线段是否相交的判断
    left_hip = Point(keypoints[8][0], keypoints[8][1])
    right_hip = Point(keypoints[11][0], keypoints[11][1])
    left_knee = Point(keypoints[9][0], keypoints[9][1])
    right_knee = Point(keypoints[12][0], keypoints[12][1])
    left_foot = Point(keypoints[10][0], keypoints[10][1])
    right_foot = Point(keypoints[13][0], keypoints[13][1])

    # 要保证所有的点都在画面中
    if left_hip.is_empty() and right_hip.is_empty() and left_knee.is_empty() \
            and right_knee.is_empty() and left_foot.is_empty() and right_foot.is_empty():
        flag1 = IsIntersec(left_hip, left_knee, right_hip, right_knee)
        flag2 = IsIntersec(left_knee, left_foot, right_knee, right_foot)
        flag3 = IsIntersec(left_hip, left_knee, right_knee, right_foot)
        flag4 = IsIntersec(left_knee, left_foot, right_hip, right_knee)
        print('IsIntersec:', flag1, flag2, flag3, flag4)
        if flag1 == 1 or flag2 == 1 or flag3 == 1 or flag4 == 1:
            # 左右大腿、左右小腿四条线之间是否相交
            labels[3] = 'unfocus2'  # 盘腿
        elif angles[2] >= 90 and angles[3] >= 90:
            # 伸腰的判断（两个腋下的角度大于某个阈值）
            labels[2] = 'unfocus1'  # 伸腰
        elif p2l_Dist > lrs_Dist / 4:
            labels[0] = 'focus1'  # 看黑板
        else:
            labels[1] = 'focus2'  # 记笔记

        return labels
    else:
        return labels


