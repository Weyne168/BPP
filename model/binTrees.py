# -*- coding: UTF-8 -*-
import copy
import pylab as pl
import torch

from model.utils import *


class rect_node(object):
    def __init__(self, rect, id=0, pid=0, left=-1, right=-1, cut_typ=-1):
        self.x, self.y, self.W, self.H = rect
        self.left = left
        self.right = right
        self.item = None
        self.cut_typ = cut_typ
        self.id = id
        self.pid = pid


class BinTree(object):
    """
    因为从左上开始排版, 所以掰边只可能发生在右边界和下边界
    """

    def __init__(self, _size, bin_type, edge=0):
        self.size = _size
        self.bin_type = bin_type
        self.edge = edge
        root = rect_node((0, 0, _size[0], _size[1]))
        self.node_list = [root]

    def length(self):
        return len(self.node_list)

    def get_ch_num(self):
        k = 0
        for i in range(len(self.node_list)):
            nd = self.node_list[i]
            if nd.left == -1 and nd.right == -1:
                k += 1
        return k

    def get_item_num(self):
        k = 0
        for i in range(len(self.node_list)):
            nd = self.node_list[i]
            if nd.left == -1 and nd.right == -1 and nd.item != None:
                k += 1
        return k

    def get_res(self):
        s = self.size[0] * self.size[1]
        for i in range(len(self.node_list)):
            nd = self.node_list[i]
            if nd.left == -1 and nd.right == -1 and nd.item != None:
                s -= nd.item[1] * nd.item[2]
        return s / (self.size[0] * self.size[1])

    def create_nodes(self, cut_pos, cut_typ, min_unit, item_list, item_mask):
        item_size = item_list * np.expand_dims(item_mask[:item_list.shape[0]] > 0, axis=1)
        min_s = np.min(item_size[:, 0] * item_size[:, 1])

        x0, y0 = cut_pos
        stack = [self.node_list[0]]
        left_rect, right_rect = None, None
        y_bias = torch.inf
        x_bias = torch.inf
        _x0 = x0
        _y0 = y0
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item == None:
                W = nd.W
                H = nd.H
                if W < min_unit or H < min_unit or W * H < min_s:
                    continue
                # 竖切,下刀处应该至少距离左边界n个单位(n应该为成品中最小尺寸),
                if cut_typ == 2 and x0 >= nd.x + min_unit and nd.x + W > x0 and abs(y0 - nd.y) < y_bias:  # 竖切
                    '''
                    if nd.x + W - x0 < min_unit:
                        continue
                    '''
                    left_rect = [nd.x, nd.y, x0 - nd.x, H]
                    right_rect = [x0, nd.y, nd.x + W - x0, H]
                    y_bias = abs(y0 - nd.y)
                    _y0 = nd.y
                    nid = nd.id

                elif cut_typ == 1 and y0 >= nd.y + min_unit and nd.y + H > y0 and abs(x0 - nd.x) < x_bias:  # 横切
                    '''
                    if nd.y + H - y0 < min_unit:
                        continue
                    '''
                    left_rect = [nd.x, nd.y, W, y0 - nd.y]
                    right_rect = [nd.x, y0, W, nd.y + H - y0]
                    x_bias = abs(x0 - nd.x)
                    _x0 = nd.x
                    nid = nd.id
                else:
                    continue

        if left_rect != None:
            left_node = rect_node(left_rect, len(self.node_list), nid)
            self.node_list.append(left_node)
            right_node = rect_node(right_rect, len(self.node_list), nid)
            self.node_list.append(right_node)
            self.node_list[nid].left = left_node.id
            self.node_list[nid].right = right_node.id
            self.node_list[nid].cut_typ = cut_typ
            return nid
        return -1

    def cut_seq_label(self):
        strips = []
        stack = [self.node_list[0]]
        # W, H = self.node_list[0].W, self.node_list[0].H
        # 中序遍历,先左后右
        strips.append([self.bin_type + 1, 0, 0, 0, 0, 0, 0, 0])
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.right != -1:
                stack.append(self.node_list[nd.right])

            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            if nd.left == -1 and nd.right == -1:
                # strips.append([1, 0, 0, 0, '0', 0, '0', 0])
                continue
            else:
                # x0, y0 = self.node_list[nd.right].x / W, self.node_list[nd.right].y / H
                x0, y0 = self.node_list[nd.right].x, self.node_list[nd.right].y
                ct = nd.cut_typ
                if self.node_list[nd.left].item == None or nd.left == -1:
                    lid = 0
                    l_rotate = 0
                else:
                    lid = self.node_list[nd.left].item[0]
                    l_rotate = self.node_list[nd.left].item[3]

                if self.node_list[nd.right].item == None or nd.right == -1:
                    rid = 0
                    r_rotate = 0
                else:
                    rid = self.node_list[nd.right].item[0]
                    r_rotate = self.node_list[nd.right].item[3]
                strips.append([0, x0, y0, ct, lid, l_rotate, rid, r_rotate])
        # 按子树最小深度从小到大排序,根据一刀切, 深度小的group先切(同样深度,靠左靠下先切,即左子树优先)
        # strips = sorted(strips, key=lambda s: s[0])
        return strips

    # 共同父节点,必须左子节点放了item,再放右子节点
    def travel_put(self, item_mask, item_list):
        # 直接放
        for i in range(len(self.node_list)):
            nd = self.node_list[i]
            if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item == None:
                pd = self.node_list[nd.pid]
                # 当前的空子节点是个左子树 or 虽然当前空子节点是右子树, 但其左兄弟不是空节点
                if pd.left == nd.id or (self.node_list[pd.left].item != None and pd.left != -1):
                    W, H = nd.W, nd.H
                    p = 0
                    S = W * H
                    candidate = -1
                    for j in range(item_mask.shape[0]):
                        if item_mask[j] > 0:
                            w, h = item_list[j, :2]
                            if (w <= W and h <= H) or (h <= W and w <= H):
                                if w * h / S > p:
                                    candidate = j
                                    p = w * h / S

                    if candidate > 0:
                        w, h = item_list[candidate, :2]
                        is_rotated = 1
                        if (w <= W and h <= H) and (h <= W and w <= H):
                            k1 = max(w / W, h / H)
                            k2 = max(h / W, w / H)
                            if k1 < k2:
                                is_rotated = 2
                        elif h <= W and w <= H:
                            is_rotated = 2

                        item = [candidate, w, h, is_rotated]
                        self.node_list[nd.id].item = item
                        item_mask[candidate] -= 1
        return item_mask

    def init_put(self, it_id, w, h):
        k1 = max(w / self.size[0], h / self.size[1])
        k2 = max(h / self.size[0], w / self.size[1])
        ww = w
        hh = h
        is_rotated = 1
        if k1 < k2:
            is_rotated = 2
            ww = h
            hh = w
        if ww / self.size[0] > hh / self.size[1]:  # 横切
            ct = 0
            left_rect = [0, 0, self.size[0], hh]
            right_rect = [0, hh, self.size[0], self.size[1] - hh]
        else:
            ct = 1
            left_rect = [0, 0, ww, self.size[1]]
            right_rect = [ww, 0, self.size[0] - ww, self.size[1]]

        left_node = rect_node(left_rect, len(self.node_list), 0)
        left_node.item = [it_id, w, h, is_rotated]
        self.node_list.append(left_node)
        right_node = rect_node(right_rect, len(self.node_list), 0)
        self.node_list.append(right_node)
        self.node_list[0].left = left_node.id
        self.node_list[0].right = right_node.id
        self.node_list[0].cut_typ = ct

    def put_item_by_id(self, nid, item_mask, item_list):
        nd = self.node_list[nid]
        X, Y, W, H = nd.x, nd.y, nd.W, nd.H
        p = 0
        S = W * H
        candidate = -1
        item_trys = []
        for i in range(len(item_mask)):
            if item_mask[i] > 0:
                w, h = item_list[i, :2]

                if w <= W and h <= H:
                    if X + w + self.edge > self.size[0] and X + w != self.size[0]:
                        continue
                    if Y + h + self.edge > self.size[1] and Y + h != self.size[1]:
                        continue

                    if w * h / S > p:
                        candidate = i
                        p = w * h / S
                        continue

                if h <= W and w <= H:
                    if X + h + self.edge > self.size[0] and X + h != self.size[0]:
                        continue
                    if Y + w + self.edge > self.size[1] and Y + w != self.size[1]:
                        continue

                    if w * h / S > p:
                        candidate = i
                        p = w * h / S

        if candidate >= 0:
            w, h = item_list[candidate, :2]
            if (w <= W and h <= H) and (X + w + self.edge <= self.size[0] or X + w == self.size[0]) and (
                    Y + h + self.edge <= self.size[1] or Y + h == self.size[1]):
                left_h = H - h
                left_w = W - w
                item_trys.append([candidate + 1, w, h, 1, left_h, left_w])

            if (h <= W and w <= H) and (X + h + self.edge <= self.size[0] or X + h == self.size[0]) and (
                    Y + w + self.edge <= self.size[1] or Y + w == self.size[1]):
                left_h = H - w
                left_w = W - h
                item_trys.append([candidate + 1, w, h, 2, left_h, left_w])
        return item_trys

    def put_items_by_pid(self, pid, item_mask, item_list):
        left = self.node_list[pid].left
        left_item_trys = self.put_item_by_id(left, item_mask, item_list)
        right = self.node_list[pid].right
        right_item_trys = self.put_item_by_id(right, item_mask, item_list)
        return left_item_trys, right_item_trys

    def put_item(self, nid, item_id, item_size, is_rotated):
        nd = self.node_list[nid]
        X, Y, W, H = nd.x, nd.y, nd.W, nd.H
        if is_rotated == 1:
            w = item_size[0]
            h = item_size[1]
        else:
            w = item_size[1]
            h = item_size[0]

        if (w <= W and h <= H) and (X + w + self.edge <= self.size[0] or X + w == self.size[0]) and (
                Y + h + self.edge <= self.size[1] or Y + h == self.size[1]):
            self.node_list[nid].item = [item_id, item_size[0], item_size[1], is_rotated]
            return True
        return False

    def tune_tree_change(self):
        stack = [self.node_list[0]]
        while len(stack) > 0:
            nd = stack.pop(-1)
            if self.node_list[nd.left].left == -1 and self.node_list[nd.left].right == -1 and self.node_list[
                nd.left].item == None:
                # 由于左下优先,所以要确保左子树深度大于等于右子树
                # 需要左右子树(上下)交换位置
                if nd.cut_typ == 1:  # 横切
                    right_W, right_H = self.get_used_size(self.node_list[nd.right])
                    old_ly = self.node_list[nd.left].y
                    ry = old_ly + right_H

                    # 子树左右交换
                    self.node_list[nd.left].y = ry
                    self.node_list[nd.left].H = nd.H - right_H
                    self.node_list[nd.right].y = old_ly
                    self.node_list[nd.right].H = right_H

                    old_left = nd.left
                    self.node_list[nd.id].left = nd.right
                    self.node_list[nd.id].right = old_left

                    self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                    self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

                else:  # 竖切
                    right_W, right_H = self.get_used_size(self.node_list[nd.right])
                    old_lx = self.node_list[nd.left].x
                    rx = old_lx + right_W

                    # 子树左右交换
                    self.node_list[nd.left].x = rx
                    self.node_list[nd.left].W = nd.W - right_W
                    self.node_list[nd.right].x = old_lx
                    self.node_list[nd.right].W = right_W

                    old_left = nd.left
                    self.node_list[nd.id].left = nd.right
                    self.node_list[nd.id].right = old_left

                    self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                    self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

    def tune_space(self):
        stack = [self.node_list[0]]
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.left != -1 and nd.right != -1 and nd.item == None:
                left_W, left_H = self.get_used_size(self.node_list[nd.left])
                if nd.cut_typ == 1:  # 横切
                    if self.node_list[nd.left].H > left_H:
                        ry = self.node_list[nd.left].y + left_H
                        self.node_list[nd.left].H = left_H

                        self.node_list[nd.right].y = ry
                        self.node_list[nd.right].H = nd.H - left_H

                        self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                        self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

                else:  # 竖切
                    if self.node_list[nd.left].W > left_W:
                        rx = self.node_list[nd.left].x + left_W
                        self.node_list[nd.left].W = left_W

                        self.node_list[nd.right].x = rx
                        self.node_list[nd.right].W = nd.W - left_W

                        self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                        self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

    def get_used_size(self, nd):
        if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item != None:
            ww, hh, r = nd.item[1:4]
            if r == 2:
                w = hh
                h = ww
            else:
                w = ww
                h = hh
            return w, h

        elif nd.cut_typ == 1 and nd.left != -1 and nd.right != -1:
            lw, lh = self.get_used_size(self.node_list[nd.left])
            rw, rh = self.get_used_size(self.node_list[nd.right])
            return max(lw, rw), lh + rh

        elif nd.cut_typ == 2 and nd.left != -1 and nd.right != -1:
            lw, lh = self.get_used_size(self.node_list[nd.left])
            rw, rh = self.get_used_size(self.node_list[nd.right])
            return lw + rw, max(lh, rh)
        return 0, 0

    def tune_bbox_xy(self, local_root, cut_typ):
        stack = [local_root]
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.item != None:
                continue

            if nd.left != -1 and nd.right != -1 and nd.item == None:
                x, y, ww, hh = nd.x, nd.y, nd.W, nd.H
                if cut_typ == 2:  # # 子树父节点竖切,只需要调整x方向上的坐标
                    self.node_list[nd.left].x = x
                    if nd.cut_typ == 2:  # 当前节点是竖切,right要调整x
                        left_W, left_H = self.get_used_size(self.node_list[nd.left])
                        self.node_list[nd.right].x = x + left_W
                        self.node_list[nd.right].W = ww - left_W
                        self.node_list[nd.left].W = left_W
                    else:
                        self.node_list[nd.right].x = x
                        self.node_list[nd.right].W = ww
                        self.node_list[nd.left].W = ww

                else:  # #子树父节点横切,只需要调整y方向上的坐标
                    self.node_list[nd.left].y = y
                    if nd.cut_typ == 2:  # 当前节点是竖切,right要调整y
                        self.node_list[nd.right].y = y
                        self.node_list[nd.right].H = hh
                        self.node_list[nd.left].H = hh
                    else:
                        left_W, left_H = self.get_used_size(self.node_list[nd.left])
                        self.node_list[nd.right].y = y + left_H
                        self.node_list[nd.right].H = hh - left_H
                        self.node_list[nd.left].H = left_H

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

    def display(self, prefix, pth='../checkpoints'):
        # cur_pth = os.getcwd()
        width, height = self.size
        bs = width * height

        colors = ['yellow', 'pink', 'blue', 'red', 'green', 'orange', 'purple']
        ax = pl.subplot(111)
        rectangle = pl.Rectangle((0, 0), width, height, color='lightgray')
        ax.add_patch(rectangle)

        s = 0
        stack = [self.node_list[0]]
        is_visited = np.zeros(len(self.node_list))
        while len(stack):
            nd = stack.pop(-1)

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            if nd.item == None and nd.id != 0:
                pd = self.node_list[nd.pid]
                if is_visited[pd.id] == 1:
                    continue
                is_visited[pd.id] = 1
                x, y, W, H = pd.x, pd.y, pd.W, pd.H
                lw, lh = self.node_list[pd.left].W, self.node_list[pd.left].H
                if pd.cut_typ == 1:  # 横切
                    xx = [x, x + lw]
                    yy = [y + lh, y + lh]
                    ax.plot(xx, yy, linestyle='--', color=colors[3])
                    # ax.add_patch(cut_line)
                else:
                    xx = [x + lw, x + lw]
                    yy = [y, y + lh]
                    ax.plot(xx, yy, linestyle='--', color=colors[3])
                    # ax.add_l(cut_line)

            elif nd.item != None:
                item_typ, w, h, is_rotated = nd.item
                s += w * h

                if is_rotated == 1:
                    ww = w
                    hh = h
                else:
                    ww = h
                    hh = w

                # color = i
                x, y = nd.x, nd.y
                rectangle = pl.Rectangle((x, y), ww, hh, color=colors[0])
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y), 1, hh, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y), ww, 1, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x + ww, y), 1, hh, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y + hh), ww, 1, color='black')
                ax.add_patch(rectangle)

                pl.text(x=x + ww / 3, y=y + hh / 3,
                        s='%d,%d' % (item_typ, is_rotated),
                        fontdict={'fontsize': 7, 'style': "italic"})

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        pl.savefig(os.path.join(pth, '%d_bin_%d_r_%.3f.png' % (prefix, self.bin_type, s / bs)))
        pl.close()
        print(os.path.join(pth, '%d_bin_%d_r_%.3f.png' % (prefix, self.bin_type, s / bs)))


class ExpBinTree(object):
    def __init__(self, bin_info):
        self.root_rect = (0, 0, bin_info[2], bin_info[3])
        root = rect_node(self.root_rect)
        self.node_list = [root]

    def create_nodes(self, cut_route):
        x0, y0, x1, y1, ww, hh = cut_route
        '''
        x0 = _x0 if _x0 <= _x1 else _x1
        y0 = _y0 if _y0 <= _y1 else _y1
        x1 = _x1 if _x0 <= _x1 else _x0
        y1 = _y1 if _y0 <= _y1 else _y0

        ww = x1 - x0
        hh = y1 - y0
        if ww == 0 and x0 == 0:  # 在最左边竖切
            return True

        if hh == 0 and y0 == 0:  # 在最下边横切
            return True
        '''

        stack = [self.node_list[0]]
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            # 叶节点可切割
            if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item == None:
                # 左和下为左节点, 右和上为右节点
                if ww == nd.W or hh == nd.H:
                    if ww == nd.W:  # 横切分上下
                        W = nd.W
                        H = y0 - nd.y
                        if nd.H < H or H < 0 or x0 != nd.x:
                            continue
                        left_rect = [nd.x, nd.y, W, H]
                        right_rect = [nd.x, y0, W, nd.H - H]
                        # print('lr', left_rect)
                        cut_typ = 0
                    else:  # 竖切分左右
                        H = nd.H
                        W = x0 - nd.x
                        if nd.W < W or W < 0 or y0 != nd.y:
                            continue
                        left_rect = [nd.x, nd.y, W, H]
                        right_rect = [nd.x + W, nd.y, nd.W - W, H]
                        cut_typ = 1

                    left_node = rect_node(left_rect, len(self.node_list), nd.id)
                    self.node_list.append(left_node)
                    right_node = rect_node(right_rect, len(self.node_list), nd.id)
                    self.node_list.append(right_node)
                    self.node_list[nd.id].left = left_node.id
                    self.node_list[nd.id].right = right_node.id

                    if self.node_list[nd.left].W == 0 or self.node_list[nd.left].H == 0:
                        self.node_list[nd.id].cut_typ = -1
                    else:
                        self.node_list[nd.id].cut_typ = cut_typ
                    return True
        return False

    def sort_cut_routes(self, cut_routes):
        cuts = []
        idx = 0
        for i in range(len(cut_routes)):
            _x0, _y0, _x1, _y1 = cut_routes[i]
            x0 = _x0 if _x0 <= _x1 else _x1
            y0 = _y0 if _y0 <= _y1 else _y1
            x1 = _x1 if _x0 <= _x1 else _x0
            y1 = _y1 if _y0 <= _y1 else _y0

            ww = x1 - x0
            hh = y1 - y0
            if ww == 0 and x0 == 0:  # 在最左边竖切
                continue

            if hh == 0 and y0 == 0:  # 在最下边横切
                continue
            cuts.append((idx, x0, y0, x1, y1, ww, hh))
            idx += 1
        cuts_h = sorted(cuts, key=lambda s: s[-1], reverse=True)
        cuts_w = sorted(cuts, key=lambda s: s[-2], reverse=True)

        tmp = [cuts_h[0]]
        b = 0
        for i in range(1, len(cuts_h)):
            if tmp[-1][-1] != cuts_h[i][-1]:
                tmp = sorted(tmp, key=lambda s: s[2])
                cuts_h[b:i] = copy.deepcopy(tmp)
                b = i
                tmp = [cuts_h[i]]
            else:
                tmp.append(cuts_h[i])

        tmp = [cuts_w[0]]
        b = 0
        for i in range(1, len(cuts_w)):
            if tmp[-1][-2] != cuts_w[i][-2]:
                tmp = sorted(tmp, key=lambda s: s[1])
                cuts_w[b:i] = copy.deepcopy(tmp)
                b = i
                tmp = [cuts_w[i]]
            else:
                tmp.append(cuts_w[i])

        # 先确定第一阶段是横切还是竖切
        ch0 = cuts_h[0][-1]
        cw0 = cuts_w[0][-2]
        is_ch = False
        if ch0 >= self.root_rect[-1]:  # 以竖切开始
            is_ch = True
        elif cw0 >= self.root_rect[-2]:  # 以横切开始
            is_ch = False
        else:
            print('error sort cuts')
        sorted_cuts = []
        rects = []
        while len(cuts_h) > 0 or len(cuts_w) > 0:
            if is_ch:  # 竖切
                is_ch = False
                if cuts_h[0][-2] != 0:
                    continue
                tmp_cuts = [cuts_h.pop(0)]

                if len(cuts_h) > 0:
                    while tmp_cuts[-1][-1] == cuts_h[0][-1] and tmp_cuts[-1][2] == cuts_h[0][2] and cuts_h[0][-2] == 0:
                        tmp_cuts.append(cuts_h.pop(0))
                        if len(cuts_h) == 0:
                            break

                tmp_cuts = sorted(tmp_cuts, key=lambda s: s[1])
                sorted_cuts.extend(copy.deepcopy(tmp_cuts))

                while len(tmp_cuts) > 0:
                    t = tmp_cuts.pop(0)
                    for i in range(len(cuts_w)):
                        if t[0] == cuts_w[i][0]:
                            cuts_w.pop(i)
                            break

            else:
                is_ch = True
                if cuts_w[0][-1] != 0:
                    continue
                tmp_cuts = [cuts_w.pop(0)]
                if len(cuts_w) > 0:
                    while tmp_cuts[-1][-2] == cuts_w[0][-2] and tmp_cuts[-1][1] == cuts_w[0][1] and cuts_w[0][-1] == 0:
                        tmp_cuts.append(cuts_w.pop(0))
                        if len(cuts_w) == 0:
                            break

                tmp_cuts = sorted(tmp_cuts, key=lambda s: s[2])
                sorted_cuts.extend(copy.deepcopy(tmp_cuts))

                while len(tmp_cuts) > 0:
                    t = tmp_cuts.pop(0)
                    for i in range(len(cuts_h)):
                        if t[0] == cuts_h[i][0]:
                            cuts_h.pop(i)
                            break
        return sorted_cuts

    def gen_tree(self, bin):
        item_list = copy.deepcopy(bin)
        cut_routes = []
        for i in range(len(item_list)):
            item = item_list[i][0]
            cut_routes.append(item[8])
        sorted_cuts = self.sort_cut_routes(cut_routes)
        e_times = 3 * len(sorted_cuts)
        while len(sorted_cuts) > 0:
            # item = item_list[i][0]
            cut_route = sorted_cuts.pop(0)
            if self.create_nodes(cut_route[1:]):
                continue
            else:
                if e_times == 0:
                    print('error case in gen')
                    return False
                sorted_cuts.append(cut_route)
                e_times -= 1
        return True

    def pack_item_into_node(self, bin):
        item_list = copy.deepcopy(bin)
        # item = [item_id, left_x, down_y, ww, hh, is_rotate, cut_typ, bin_typ, (cut_route[0]['x'], cut_route[0]['y']), (W, H, edge)])
        for i in range(len(bin)):
            item = item_list[i][0]
            x, y, ww, hh, r = item[1:6]
            if r == 1:
                t = ww
                ww = hh
                hh = t
            k = 0
            pos = -1
            for j in range(len(self.node_list)):
                nd = self.node_list[j]
                if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item == None:
                    p, s = iou((x, y, ww, hh), (nd.x, nd.y, nd.W, nd.H))
                    # print(s, ww * hh)
                    if s == ww * hh and p > k:
                        k = p
                        pos = j
            if pos != -1:
                self.node_list[pos].item = item
            else:
                print('errr case!!!')
                return False
        return True

    def get_used_size(self, nd):
        if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item != None:
            ww, hh, r = nd.item[3:6]
            if r == 1:
                w = hh
                h = ww
            else:
                w = ww
                h = hh
            return w, h

        elif nd.cut_typ == 0 and nd.left != -1 and nd.right != -1:
            lw, lh = self.get_used_size(self.node_list[nd.left])
            rw, rh = self.get_used_size(self.node_list[nd.right])
            return max(lw, rw), lh + rh

        elif nd.cut_typ == 1 and nd.left != -1 and nd.right != -1:
            lw, lh = self.get_used_size(self.node_list[nd.left])
            rw, rh = self.get_used_size(self.node_list[nd.right])
            return lw + rw, max(lh, rh)
        return 0, 0

    def tune_bbox_xy(self, local_root, cut_typ):
        stack = [local_root]
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.item != None:
                continue

            x, y, ww, hh = nd.x, nd.y, nd.W, nd.H
            if cut_typ == 1:  #
                self.node_list[nd.left].x = x
                if nd.cut_typ == 1:  # 当前节点是竖切,right要调整x
                    left_W, left_H = self.get_used_size(self.node_list[nd.left])
                    self.node_list[nd.right].x = x + left_W
                    self.node_list[nd.right].W = ww - left_W
                    self.node_list[nd.left].W = left_W
                else:
                    self.node_list[nd.right].x = x
                    self.node_list[nd.right].W = ww
                    self.node_list[nd.left].W = ww
            else:
                self.node_list[nd.left].y = y
                if nd.cut_typ == 1:  # 当前节点是竖切,right要调整y
                    self.node_list[nd.right].y = y
                    self.node_list[nd.right].H = hh
                    self.node_list[nd.left].H = hh
                else:
                    left_W, left_H = self.get_used_size(self.node_list[nd.left])
                    self.node_list[nd.right].y = y + left_H
                    self.node_list[nd.right].H = hh - left_H
                    self.node_list[nd.left].H = left_H

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

    def tune_tree_depth(self):
        stack = [self.node_list[0]]
        t = 0
        while len(stack) > 0:
            nd = stack.pop(-1)

            left_depth = self.get_tree_depth(nd.left)
            right_depth = self.get_tree_depth(nd.right)

            # left_depth = self.get_min_depth(nd.left)
            # right_depth = self.get_min_depth(nd.right)

            # 交换左右子树,确保左子树深度小于等于右子树
            if nd.cut_typ == 0:  # 横切
                if left_depth > right_depth:  # 需要左右子树(上下)交换位置
                    right_W, right_H = self.get_used_size(self.node_list[nd.right])
                    old_ly = self.node_list[nd.left].y
                    ry = old_ly + right_H

                    # 子树左右交换
                    self.node_list[nd.left].y = ry
                    self.node_list[nd.left].H = nd.H - right_H
                    self.node_list[nd.right].y = old_ly
                    self.node_list[nd.right].H = right_H

                    old_left = nd.left
                    self.node_list[nd.id].left = nd.right
                    self.node_list[nd.id].right = old_left

                    self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                    self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

                    t += 1

            else:  # 竖切
                if left_depth > right_depth:
                    right_W, right_H = self.get_used_size(self.node_list[nd.right])
                    old_lx = self.node_list[nd.left].x
                    rx = old_lx + right_W

                    # 子树左右交换
                    self.node_list[nd.left].x = rx
                    self.node_list[nd.left].W = nd.W - right_W
                    self.node_list[nd.right].x = old_lx
                    self.node_list[nd.right].W = right_W

                    old_left = nd.left
                    self.node_list[nd.id].left = nd.right
                    self.node_list[nd.id].right = old_left

                    self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                    self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

                    t += 1

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])
        return t

    def tune_tree_cut(self):
        stack = [self.node_list[0]]
        t = 0
        while len(stack) > 0:
            nd = stack.pop(-1)
            pd = self.node_list[nd.pid]
            '''

            while (pd.cut_typ == nd.cut_typ) and pd.id != 0:
                pd = self.node_list[pd.pid]
            '''
            if nd.left == -1 or nd.right == -1:
                continue

            if pd.id == nd.id:
                if nd.right != -1:
                    stack.append(self.node_list[nd.right])
                if nd.left != -1:
                    stack.append(self.node_list[nd.left])
                continue

            # 左下优先排版,因此总是先切左子树的group
            if pd.cut_typ != nd.cut_typ:
                left_W, left_H = self.get_used_size(self.node_list[nd.left])
                right_W, right_H = self.get_used_size(self.node_list[nd.right])
                if pd.cut_typ == 0:  # 当前节点的父节点是横切,要求当前节点的左子树排版的used_size的高>=右子树
                    if right_H > left_H:  # 当前节点cut_typ==1, 需要左右子树(左右)交换位置
                        old_lx = self.node_list[nd.left].x
                        rx = old_lx + right_W

                        # 子树左右交换放置的位置
                        self.node_list[nd.left].x = rx
                        self.node_list[nd.left].W = nd.W - right_W
                        self.node_list[nd.right].x = old_lx
                        self.node_list[nd.right].W = right_W

                        old_left = nd.left
                        self.node_list[nd.id].left = nd.right
                        self.node_list[nd.id].right = old_left

                        self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                        self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)
                        t += 1

                else:  # # 当前节点的父节点是竖切,要求当前节点的左子树排版的used_size的left_W>=right_W
                    if right_W > left_W:  # 当前节点cut_typ==0, 需要左右子树(上下)交换位置
                        old_ly = self.node_list[nd.left].y
                        ry = old_ly + right_H

                        # 子树左右交换
                        self.node_list[nd.left].y = ry
                        self.node_list[nd.left].H = nd.H - right_H
                        self.node_list[nd.right].y = old_ly
                        self.node_list[nd.right].H = right_H

                        old_left = nd.left
                        self.node_list[nd.id].left = nd.right
                        self.node_list[nd.id].right = old_left

                        self.tune_bbox_xy(self.node_list[nd.left], nd.cut_typ)
                        self.tune_bbox_xy(self.node_list[nd.right], nd.cut_typ)

                        t += 1

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])
        return t

    def get_group_attributes(self, group):
        # pass
        cut_typ_in = group[0][-1]
        bx, by, W, H = group[1]

        items, bws, bhs = [], [], []
        for i in range(2, len(group)):
            nd = group[i]
            if nd.item == None:
                print('eeerrr')
                exit(1)
            it = nd.item
            if it[5] == 1:
                bws.append(it[4])
                bhs.append(it[3])
            else:
                bws.append(it[3])
                bhs.append(it[4])
            items.append(it)

        if len(group[0]) > 1:
            if cut_typ_in == 0:  # 横切
                bh = sum(bhs)
                bw = W
            else:
                bw = sum(bws)
                bh = H
        else:
            bw = W
            bh = H

        items = sorted(items, key=lambda s: s[3] * s[4], reverse=True)
        group_info = [group[0], [bx, by, bw, bh]]
        group_info.extend(items)
        return group_info

    def get_node_depth(self, node):
        nd = copy.deepcopy(node)
        n = 0
        while nd.pid != 0:
            nd = self.node_list[nd.pid]
            n += 1
        return n

    def get_tree_depth(self, root):
        if self.node_list[root].left == -1 and self.node_list[root].right == -1:
            return 1
        else:
            if self.node_list[root].left != -1:
                left_depth = self.get_tree_depth(self.node_list[root].left)
            else:
                left_depth = 0

            if self.node_list[root].right != -1:
                right_depth = self.get_tree_depth(self.node_list[root].right)
            else:
                right_depth = 0
            return max(left_depth, right_depth) + 1

    def get_min_depth(self, root):
        if self.node_list[root].left == -1 and self.node_list[root].right == -1:
            return 1
        else:
            if self.node_list[root].left != -1:
                left_depth = self.get_min_depth(self.node_list[root].left)
            else:
                left_depth = 0

            if self.node_list[root].right != -1:
                right_depth = self.get_min_depth(self.node_list[root].right)
            else:
                right_depth = 0
            return min(left_depth, right_depth) + 1

    def split_sub_trees(self):
        strips = []
        stack = [self.node_list[0]]
        is_visited = np.zeros(len(self.node_list))
        # 遍历树,先左后右
        while len(stack) > 0:
            nd = stack.pop(-1)

            if nd.right != -1:
                stack.append(self.node_list[nd.right])

            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            # 找到了叶子节点
            if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and is_visited[
                nd.id] == 0 and nd.item != None:
                pd = self.node_list[nd.pid]
                cut_typ_in = pd.cut_typ
                group_cut = [cut_typ_in]

                # 找出group_out节点(cut_typ改变or 左右子树cut_typ不一样的节点)
                local_root = pd
                is_left = 1
                while (cut_typ_in == local_root.cut_typ) and local_root.id != 0:  # and is_visited[local_root.id] == 0:
                    if self.node_list[local_root.pid].left == local_root.id:
                        is_left = 1
                    else:
                        is_left = 0
                    local_root = self.node_list[local_root.pid]

                _local_root = local_root
                _is_left = 1
                while _local_root.cut_typ != cut_typ_in and is_visited[_local_root.id] == 1 and _local_root.id != 0:
                    if self.node_list[_local_root.pid].left == _local_root.id:
                        _is_left = 1
                    else:
                        _is_left = 0
                    _local_root = self.node_list[_local_root.pid]

                if is_visited[_local_root.id] == 0 and local_root.id != _local_root.id:
                    local_root = _local_root
                    is_left = _is_left

                if local_root.id == 0 and local_root.cut_typ == cut_typ_in:
                    is_visited[local_root.id] = 1

                if is_visited[local_root.id] == 0 and local_root.cut_typ != cut_typ_in:
                    group_cut.insert(0, local_root.cut_typ)
                    is_visited[local_root.id] = 1

                    stop = self.node_list[local_root.pid]
                    while stop.cut_typ != cut_typ_in and stop.id != 0:
                        stop = self.node_list[stop.pid]

                    if stop.cut_typ == cut_typ_in and is_visited[stop.id] == 0:
                        group_cut.insert(0, stop.cut_typ)
                        is_visited[stop.id] = 1

                if local_root.id != pd.id:
                    if is_left == 1:
                        sub_root = self.node_list[local_root.left]
                    else:
                        sub_root = self.node_list[local_root.right]
                else:
                    sub_root = pd
                local_stack = [sub_root]
                # is_visited[sub_root.id] = 1

                group = [group_cut, [sub_root.x, sub_root.y, sub_root.W, sub_root.H]]
                while len(local_stack) > 0:
                    d = local_stack.pop(-1)
                    if d.right != -1:
                        if self.node_list[d.right].cut_typ == cut_typ_in or self.node_list[d.right].item != None:
                            local_stack.append(self.node_list[d.right])
                    if d.left != -1:
                        if self.node_list[d.left].cut_typ == cut_typ_in or self.node_list[d.left].item != None:
                            local_stack.append(self.node_list[d.left])

                    if d.left == -1 and d.right == -1 and d.W != 0 and d.H != 0 and is_visited[
                        d.id] == 0 and d.item != None:
                        group.append(d)
                        is_visited[d.id] = 1
                        is_visited[d.pid] = 1

                group = self.get_group_attributes(group)
                strips.append(group)
        # 按子树最小深度从小到大排序,根据一刀切, 深度小的group先切(同样深度,靠左靠下先切,即左子树优先)
        # strips = sorted(strips, key=lambda s: s[0])
        return strips

    def cut_seq_label(self, bid):
        strips = []
        stack = [self.node_list[0]]
        # W, H = self.node_list[0].W, self.node_list[0].H
        # 中序遍历,先左后右
        strips.append([bid + 2, 0, 0, 0, '0', 0, '0', 0])
        while len(stack) > 0:
            nd = stack.pop(-1)
            if nd.right != -1:
                stack.append(self.node_list[nd.right])

            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            if nd.left == -1 and nd.right == -1:
                # strips.append([1, 0, 0, 0, '0', 0, '0', 0])
                continue

            else:
                # x0, y0 = self.node_list[nd.right].x / W, self.node_list[nd.right].y / H
                x0, y0 = self.node_list[nd.right].x, self.node_list[nd.right].y
                ct = nd.cut_typ + 1
                if self.node_list[nd.left].item == None:
                    lid = '0'
                    l_rotate = 0
                else:
                    lid = self.node_list[nd.left].item[0]
                    l_rotate = self.node_list[nd.left].item[5] + 1

                if self.node_list[nd.right].item == None:
                    rid = '0'
                    r_rotate = 0
                else:
                    rid = self.node_list[nd.right].item[0]
                    r_rotate = self.node_list[nd.right].item[5] + 1
                    # bin_id==0,表示非新开片节点,1表示结束点,2开始是原片类型
                strips.append([0, x0, y0, ct, lid, l_rotate, rid, r_rotate])
        # 按子树最小深度从小到大排序,根据一刀切, 深度小的group先切(同样深度,靠左靠下先切,即左子树优先)
        # strips = sorted(strips, key=lambda s: s[0])
        return strips

    def display(self, prefix, pth='../checkpoints3'):
        # cur_pth = os.getcwd()
        width, height = self.root_rect[2:]
        bs = width * height

        colors = ['yellow', 'pink', 'blue', 'red', 'green', 'orange', 'purple']
        ax = pl.subplot(111)
        rectangle = pl.Rectangle((0, 0), width, height, color='lightgray')
        ax.add_patch(rectangle)

        s = 0
        stack = [self.node_list[0]]
        is_visited = np.zeros(len(self.node_list))
        while len(stack):
            nd = stack.pop(-1)

            if nd.right != -1:
                stack.append(self.node_list[nd.right])
            if nd.left != -1:
                stack.append(self.node_list[nd.left])

            if nd.item == None and nd.id != 0:
                pd = self.node_list[nd.pid]
                if is_visited[pd.id] == 1:
                    continue
                is_visited[pd.id] = 1
                x, y, W, H = pd.x, pd.y, pd.W, pd.H
                lw, lh = self.node_list[pd.left].W, self.node_list[pd.left].H
                if pd.cut_typ == 1:  # 横切
                    xx = [x, x + lw]
                    yy = [y + lh, y + lh]
                    ax.plot(xx, yy, linestyle='--', color=colors[3])
                    # ax.add_patch(cut_line)
                else:
                    xx = [x + lw, x + lw]
                    yy = [y, y + lh]
                    ax.plot(xx, yy, linestyle='--', color=colors[3])
                    # ax.add_l(cut_line)

            elif nd.item != None:
                w, h, is_rotated = nd.item[3:6]
                if is_rotated == 1:
                    ww = h
                    hh = w
                else:
                    ww = w
                    hh = h

                item_typ = nd.item[0]

                s += ww * hh

                # color = i
                x, y = nd.x, nd.y
                rectangle = pl.Rectangle((x, y), ww, hh, color=colors[0])
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y), 1, hh, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y), ww, 1, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x + ww, y), 1, hh, color='black')
                ax.add_patch(rectangle)

                rectangle = pl.Rectangle((x, y + hh), ww, 1, color='black')
                ax.add_patch(rectangle)

                pl.text(x=x + ww / 3, y=y + hh / 3,
                        s='%d,%d' % (item_typ, is_rotated),
                        fontdict={'fontsize': 7, 'style': "italic"})

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        pl.savefig(os.path.join(pth, '%d_bin_r_%.3f.png' % (prefix, s / bs)))
        pl.close()
        print(os.path.join(pth, '%d_bin_r_%.3f.png' % (prefix, s / bs)))
