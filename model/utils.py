# -*- coding: UTF-8 -*-
import os
import glob
import torch
import numpy as np


def iou(rec1, rec2):
    x1, y1, w1, h1 = rec1  # box1的左下角坐标(x1, y1)和宽高(w1, h1)
    x2, y2, w2, h2 = rec2  # box2的左下角坐标(x2, y2)和宽高(w2, h2)
    #print('r1',rec1)
    #print('r2', rec2)
    # 计算两个边界框的交集
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)

    # 计算交集面积
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # 计算并集面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou, intersection_area





def check_edge_fit(blank, item, total_space, edge):
    WW, HH = total_space
    X, Y, W, H = blank
    x, y, w, h = item

    if x < 0 or y < 0:
        return False

    if x < X or y < Y:
        return False

    if x + w > X + W or y + h > Y + H:
        return False

    if (y + h + edge <= HH or y + h == HH) and h <= H:
        pass
    else:
        return False

    if (x + w + edge <= WW or x + w == WW) and w <= W:
        pass
    else:
        return False

    return True


def iou_line(l1, l2):
    x1, x2 = l1
    y1, y2 = l2
    r = (min(x2, y2) - max(x1, y1)) / ((max(x2, y2) - min(x1, y1)) + 1e-6)
    return r


def check_guillotine(right_bank, node):
    # 当前item所在的右框内,不能存在其他item
    # rec1 = [right_bank[0], right_bank[1], right_bank[0] + right_bank[2], right_bank[1] + right_bank[3]]
    rec2 = [node.x, node.y, node.ww, node.hh]
    return check_item_cross(right_bank, rec2)


def check_space_fit(blank, space, item, total_space, edge):
    '''
    :return: True | False 是否能放下; loc_d: 预测位置与实际放置位置的距离; 实际放置位置
    '''
    WW, HH = total_space
    X, Y, W, H = blank
    x, y, w, h, is_rotated = item

    if x < X or y < Y:
        return False

    if space < w * h:
        return False

    if is_rotated == 1:
        ww = h
        hh = w
    else:
        ww = w
        hh = h

    if x + ww > X + W or y + hh > Y + H:
        return False

    # check the size
    if (y + hh + edge <= HH or y + hh == HH) and hh <= H:
        r_edge = 1
    else:
        r_edge = 0

    if (x + ww + edge <= WW or x + ww == WW) and ww <= W:
        d_edge = 1
    else:
        d_edge = 0

    # check the cut type
    if r_edge * d_edge == 1:
        return True

    return False


def cross_product(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)


def is_intersected(x1, y1, x2, y2, x3, y3, x4, y4):
    if (max(x1, x2) > min(x3, x4) and max(x3, x4) > min(x1, x2) and
            max(y1, y2) > min(y3, y4) and max(y3, y4) > min(y1, y2) and
            cross_product(x1, y1, x2, y2, x3, y3) * cross_product(x1, y1, x2, y2, x4, y4) < 0 and
            cross_product(x3, y3, x4, y4, x1, y1) * cross_product(x3, y3, x4, y4, x2, y2) < 0):
        return True
    return False


def check_cut_vertical_cross(target_node, cut_track, cut_type):
    '''
    判断刀路上是否有其他item
    '''

    Ax0 = target_node.x
    Ay0 = target_node.y

    Bx0 = target_node.x + target_node.ww
    By0 = Ay0

    Ax1 = Ax0
    Ay1 = target_node.y + target_node.hh

    Bx1 = Bx0
    By1 = Ay1

    Cx0, Cy0, Cx1, Cy1 = cut_track
    if cut_type == 0 or cut_type == 1:  # 竖切
        # (A0,B0), (A1,B1)
        # A0->C0 * A0->C1
        '''
        cp1 = (Cx0 - Ax0) * (Cx1 - Ax0) - (Cy0 - Ay0) * (Cy1 - Ay0)
        # B0->C0 * B0->C1
        cp2 = (Cx0 - Bx0) * (Cx1 - Bx0) - (Cy0 - By0) * (Cy1 - By0)

        # C0->A0 * C0->B0
        cp3 = (Ax0 - Cx0) * (Bx0 - Cx0) - (Ay0 - Cy0) * (By0 - Cy0)
        # C1->A0 * C1->B0
        cp4 = (Ax0 - Cx1) * (Bx0 - Cx1) - (Ay0 - Cy1) * (By0 - Cy1)
        
        # 若 cp1 和 cp2 的乘积小于等于零,且 cp3 和 cp4 的乘积小于等于零,这两条线段相交
        if cp1 * cp2 < 0 and cp3 * cp4 < 0:
            return True
        '''
        if is_intersected(Cx0, Cy0, Cx1, Cy1, Ax0, Ay0, Bx0, By0):
            return True
        if is_intersected(Cx0, Cy0, Cx1, Cy1, Ax1, Ay1, Bx1, By1):
            return True



    else:  # 横切
        # (A0,A1), (B0,B1)
        # A0->C0 * A0->C1
        '''
        
        cp1 = (Cx0 - Ax0) * (Cx1 - Ax0) - (Cy0 - Ay0) * (Cy1 - Ay0)

        # A1->C0 * A1->C1
        cp2 = (Cx0 - Ax1) * (Cx1 - Ax1) - (Cy0 - Ay1) * (Cy1 - Ay1)

        # C0->A0 * C0->A1
        cp3 = (Ax0 - Cx0) * (Ax1 - Cx0) - (Ay0 - Cy0) * (Ay1 - Cy0)
        # C1->A0 * C1->A1
        cp4 = (Ax0 - Cx1) * (Ax1 - Cx1) - (Ay0 - Cy1) * (Ay1 - Cy1)

        # 若 cp1 和 cp2 的乘积小于等于零,且 cp3 和 cp4 的乘积小于等于零,这两条线段相交
        if cp1 * cp2 < 0 and cp3 * cp4 < 0:
            return True
        '''
        if is_intersected(Cx0, Cy0, Cx1, Cy1, Ax0, Ay0, Ax1, Ay1):
            return True
        if is_intersected(Cx0, Cy0, Cx1, Cy1, Bx0, By0, Bx1, By1):
            return True

    return False


def check_item_cross(rec1, rec2):
    '''
    判断两个矩形框是否有重叠,true有交集
    '''
    x1, y1, w, h = rec1
    x2, y2 = x1 + w, y1 + h

    a1, b1, w, h = rec2
    a2, b2 = a1 + w, b1 + h
    '''
    
    if (a1 > x2) or (a2 < x1) or (b2 < y1) or (b1 > y2):
        return False

    '''

    x_overlap = max(0, min(x2, a2) - max(x1, a1))
    y_overlap = max(0, min(y2, b2) - max(y1, b1))
    if x_overlap * y_overlap > 0:
        return True

    return False


def files(curr_dir='.', ext='*.png'):
    """当前目录下的文件"""
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i


def remove_files(rootdir, ext='*.png'):
    """删除rootdir目录下的符合的文件"""
    for i in files(rootdir, ext):
        os.remove(i)


def soft_update_actor(target, source, tau):
    for target_param, param in zip(target.named_parameters(), source.named_parameters()):

        if target_param[0].find('critic') != -1:
            # print(target_param[0])
            continue

        if target_param[1].requires_grad:
            target_param[1].data.copy_(
                target_param[1].data * (1.0 - tau) + param[1].data * tau
            )
        else:
            target_param[1].data.copy_(param[1].data)


def soft_update_critic(target, source, tau):
    for target_param, param in zip(target.named_parameters(), source.named_parameters()):

        if target_param[0].find('critic') == -1:
            # print(target_param[0])
            continue

        if target_param[1].requires_grad:
            target_param[1].data.copy_(
                target_param[1].data * (1.0 - tau) + param[1].data * tau
            )
        else:
            target_param[1].data.copy_(param[1].data)
