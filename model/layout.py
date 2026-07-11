# -*- coding: UTF-8 -*-
import copy

import numpy as np
import pylab as pl
from model.utils import *
from model.binTrees import BinTree, rect_node


class Layout(object):
    def __init__(self, *args):
        self.constructor(*args)

    def constructor(self, material, lid, typ_id, edge=0):
        '''
        materials: [[num, W, H]]
        '''
        self.lid = lid
        self.bin_size = [material[1].item(), material[2].item()]  # material[1:]
        self.edge = edge
        self.bin_tree = BinTree(self.bin_size, typ_id, edge)
        self.bin_typ = typ_id


    def step_pack(self, display_seq, cut_typs):
        if cut_typs[-1] == 1:  # 组内上横切, 组外右竖切(or 不必切)
            H = 0
            W = np.max(np.array(display_seq)[:, 0])
            for i in range(len(display_seq)):
                H += display_seq[i][1]
        else:
            W = 0
            H = np.max(np.array(display_seq)[:, 1])
            for i in range(len(display_seq)):
                W += display_seq[i][0]
        cut_box = [W, H]
        fit_node, used_ratio = self.bin_tree.travel_check(cut_box)
        return cut_box, fit_node, used_ratio

    def put_group(self, fit_node, cut_box, tune_tree, display_seq, group_cuts):

        if tune_tree != None:
            self.bin_tree.node_list = copy.deepcopy(tune_tree)

        W, H = cut_box
        group_cut_in = group_cuts[-1]
        group_cut_out = group_cuts[-2]
        group_cut_0 = group_cuts[0]
        cur_node = self.bin_tree.node_list[fit_node]
        cut_box = [cur_node.x, cur_node.y, W, H]

        left_rect, right_rect = None, None
        if group_cut_in == 1:  # 组内上横切1, 组外右竖切2
            if group_cut_0 != 0:
                group_cut_0 = 1
                group_cut_out = 2
                left_rect = [cur_node.x, cur_node.y, cur_node.W, H]
                right_rect = [cur_node.x, cur_node.y + H, cur_node.W, cur_node.H - H]

                left_node = rect_node(left_rect, len(self.bin_tree.node_list), cur_node.id)
                self.bin_tree.node_list.append(left_node)
                right_node = rect_node(right_rect, len(self.bin_tree.node_list), cur_node.id)
                self.bin_tree.node_list.append(right_node)
                self.bin_tree.node_list[cur_node.id].left = left_node.id
                self.bin_tree.node_list[cur_node.id].right = right_node.id
                self.bin_tree.node_list[cur_node.id].cut_typ = group_cut_0
                cur_node = self.bin_tree.node_list[cur_node.left]

            if group_cut_out != 0:
                group_cut_out = 2
                left_rect = [cur_node.x, cur_node.y, W, cur_node.H]
                right_rect = [cur_node.x + W, cur_node.y, cur_node.W - W, cur_node.H]

        else:  # 组内右竖切2, 组外横上切1
            if group_cut_0 != 0:
                group_cut_0 = 2
                group_cut_out = 1
                left_rect = [cur_node.x, cur_node.y, W, cur_node.H]
                right_rect = [cur_node.x + W, cur_node.y, cur_node.W - W, cur_node.H]

                left_node = rect_node(left_rect, len(self.bin_tree.node_list), cur_node.id)
                self.bin_tree.node_list.append(left_node)
                right_node = rect_node(right_rect, len(self.bin_tree.node_list), cur_node.id)
                self.bin_tree.node_list.append(right_node)
                self.bin_tree.node_list[cur_node.id].left = left_node.id
                self.bin_tree.node_list[cur_node.id].right = right_node.id
                self.bin_tree.node_list[cur_node.id].cut_typ = group_cut_0
                cur_node = self.bin_tree.node_list[cur_node.left]

            if group_cut_out != 0:
                group_cut_out = 1
                left_rect = [cur_node.x, cur_node.y, cur_node.W, H]
                right_rect = [cur_node.x, cur_node.y + H, cur_node.W, cur_node.H - H]

        if group_cut_out != 0:
            cut_box = left_rect

            left_node = rect_node(left_rect, len(self.bin_tree.node_list), cur_node.id)
            self.bin_tree.node_list.append(left_node)
            right_node = rect_node(right_rect, len(self.bin_tree.node_list), cur_node.id)
            self.bin_tree.node_list.append(right_node)
            self.bin_tree.node_list[cur_node.id].left = left_node.id
            self.bin_tree.node_list[cur_node.id].right = right_node.id
            self.bin_tree.node_list[cur_node.id].cut_typ = group_cut_out
            cur_node = self.bin_tree.node_list[cur_node.left]

        for i in range(len(display_seq)):
            ww, hh, item_typ, is_rotated = display_seq[i]
            if group_cut_in == 1:  # 组内横切1
                left_rect = [cur_node.x, cur_node.y, cur_node.W, hh]
                right_rect = [cur_node.x, cur_node.y + hh, cur_node.W, cur_node.H - hh]
            else:
                left_rect = [cur_node.x, cur_node.y, ww, cur_node.H]
                right_rect = [cur_node.x + ww, cur_node.y, cur_node.W - ww, cur_node.H]

            left_node = rect_node(left_rect, len(self.bin_tree.node_list), cur_node.id)
            left_node.item = display_seq[i]

            s = display_seq[i][0] * display_seq[i][1]
            left_node.residue -= s
            self.bin_tree.node_list.append(left_node)
            pd = self.bin_tree.node_list[left_node.pid]
            while pd.id != 0:
                self.bin_tree.node_list[pd.id].residue -= s
                pd = self.bin_tree.node_list[pd.pid]
            self.bin_tree.node_list[0].residue -= s

            right_node = rect_node(right_rect, len(self.bin_tree.node_list), cur_node.id)
            self.bin_tree.node_list.append(right_node)
            self.bin_tree.node_list[cur_node.id].left = left_node.id
            self.bin_tree.node_list[cur_node.id].right = right_node.id
            self.bin_tree.node_list[cur_node.id].cut_typ = group_cut_in
            cur_node = self.bin_tree.node_list[cur_node.right]
            self.item_list.append([ww, hh, item_typ])
        group_info = [group_cut_0, group_cut_out, group_cut_in]
        group_info.extend(cut_box)
        return group_info

    def display(self, pth):
        self.bin_tree.display(self.lid, pth)

    def get_cut_ratio(self):
        S = self.bin_size[0] * self.bin_size[1]
        r = 1 - self.bin_tree.node_list[0].residue / S
        return r
