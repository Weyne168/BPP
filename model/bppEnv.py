# -*- coding: UTF-8 -*-
import copy
import sys

sys.path.append('..')
import pylab as pl
from model.utils import *
from model.layout import Layout
from torch.utils.data import DataLoader


class Env(object):
    def __init__(self, sheets, items, edge=0):
        self.layouts = []
        self.items = np.zeros((items.shape[0] + 1, items.shape[1]), dtype=np.int32)
        self.sheets = np.zeros((sheets.shape[0] + 1, sheets.shape[1]), dtype=np.int32)
        self.edge = edge
        # item, sheet 从1开始编号
        self.items[1:] = items[...]
        self.sheets[1:] = sheets[...]
        self.item_num = sum(self.items[1:, -1])
        self.min_unit = min(self.items[1:, 0].min(), self.items[1:, 1].min())

    def new_bin(self, bin_typ):
        bin_typ = bin_typ + 1
        N, W, H = self.sheets[bin_typ]
        if N != 0:
            layout = Layout(self.sheets[bin_typ], len(self.layouts), bin_typ, self.edge)
            self.layouts.append(layout)
            self.cur_bin_typ = bin_typ
            return True
        return False

    def remove_blank_layouts(self):
        lays = []
        while len(self.layouts) > 0:
            lay = self.layouts.pop(0)
            if lay.bin_tree.get_item_num() > 0:
                lays.append(lay)
        self.layouts = lays

    def get_env_strips(self):
        strips = []
        for i in range(len(self.layouts)):
            layout = self.layouts[i]
            s = layout.bin_tree.cut_seq_label()
            strips.extend(s)
        return strips

    def get_put_item_num(self):
        k = 0
        for i in range(len(self.layouts)):
            layout = self.layouts[i]
            k += layout.bin_tree.get_item_num()
        return k

    def get_max_size_item(self, item_mask):
        for i in range(1, self.items.shape[0]):
            if item_mask[0, i] > 0:
                return i, self.items[i, 0], self.items[i, 1]
        return i, 0, 0

    def pack_items(self, display_seq, cut_typs):
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

        return cut_box

    def step_group(self, group_seq):
        seq = copy.deepcopy(group_seq)
        if seq[0][0] == 1:
            items = seq[2:]
            group_cuts = seq[1][2:5]
        else:
            items = seq[1:]
            group_cuts = seq[0][2:5]

        display_seq = []
        item_mask = copy.deepcopy(self.items[:, -1])
        for i in range(len(items)):
            _, _, _, _, _, item_typ, item_rt = items[i]
            n = item_mask[item_typ]
            if n != 0:
                w, h, _ = self.items[item_typ]
                if item_rt == 2:  # 0为空,1不旋转,2旋转
                    ww = h
                    hh = w
                else:
                    ww = w
                    hh = h
                display_seq.append([ww, hh, item_typ, item_rt])
                item_mask[item_typ] -= 1

        best_fit_node = -1
        least_used_ratio = 0
        best_fit_layout = -1
        best_display_seq = None
        best_cut_box = None
        best_tune_tree = None
        while best_fit_layout == -1 and len(display_seq) > 0:
            cut_box = self.pack_items(display_seq, group_cuts)
            for i in range(len(self.layouts) - 1, -1, -1):
                if self.layouts[i].bin_typ != self.cur_bin_typ:
                    continue

                fit_node, used_ratio, tune_tree = self.layouts[i].bin_tree.travel_check(cut_box)
                if used_ratio > least_used_ratio:
                    best_fit_layout = i
                    best_fit_node = fit_node
                    least_used_ratio = used_ratio
                    best_tune_tree = tune_tree
                    best_cut_box = copy.deepcopy(cut_box)
                    best_display_seq = copy.deepcopy(display_seq)

            # 放不下
            if best_fit_layout == -1:
                display_seq.pop()

        new_seq = []
        if best_fit_layout != -1:
            group_cut_info = self.layouts[best_fit_layout].put_group(best_fit_node, best_cut_box, best_tune_tree,
                                                                     best_display_seq,
                                                                     group_cuts)
            if seq[0][0] == 1:
                new_seq.append(seq[0])
            new_seq.append([2, 0, group_cut_info[0], group_cut_info[1], group_cut_info[2], 0, 0])

            for i in range(len(display_seq)):
                self.items[display_seq[i][-2], -1] -= 1
                new_seq.append([3, 0, 0, 0, 0, display_seq[i][-2], display_seq[i][-1]])
        return new_seq

    def get_cut_ratio(self):
        bin_area = 0
        item_area = 0
        for layout in self.layouts:
            bin_area += layout.bin_size[0] * layout.bin_size[1]
            for i in range(len(layout.bin_tree.node_list)):
                if layout.bin_tree.node_list[i].item != None:
                    it_id = layout.bin_tree.node_list[i].item[0]
                    w, h, _ = self.items[it_id]
                    item_area += w * h
        return item_area / bin_area

    def display(self, name, pth='../checkpoints2'):
        layouts = self.layouts
        pth = os.path.join(pth, name)
        print(len(layouts), pth)

        if os.path.exists(pth):
            remove_files(pth)
        else:
            os.makedirs(pth)
        num = 0
        for i in range(len(layouts)):
            layouts[i].display(pth)
            num += layouts[i].bin_tree.get_item_num()
        print('num===', num)

    def layout_filter(self, idx, dataset):
        filter_data = []
        filter_layout = []
        is_filer = False
        remove_num = 0
        for i in range(len(self.layouts)):
            if abs(dataset.lab_item_seq[idx][i][1] - self.layouts[i].get_cut_ratio()) > 0.001:
                remove_num += 1
                bin = dataset.lab_item_seq[idx][i]
                is_filer = True
                for g in range(len(bin[0])):
                    items = bin[0][g][2:]
                    for k in range(len(items)):
                        tid = items[k][0]
                        dataset.total_order_list[idx][tid, -1] -= 1
            else:
                filter_data.append(dataset.lab_item_seq[idx][i])
                filter_layout.append(self.layouts[i])
        self.layouts = copy.deepcopy(filter_layout)
        dataset.lab_item_seq[idx] = copy.deepcopy(filter_data)
        return is_filer, remove_num

    def layout_filter2(self, idx, dataset):
        filter_data = []
        filter_layout = []
        is_filer = False
        for i in range(len(self.layouts)):
            if abs(dataset.lab_item_seq[idx][i][1] - self.layouts[i].get_cut_ratio()) > 0.001:
                dataset.lab_item_seq.pop(idx)
                dataset.total_order_list.pop(idx)
                dataset.total_bins_list.pop(idx)
                dataset.total_edge_list.pop(idx)
                dataset.cut_ratio.pop(idx)
                is_filer = True
                break

        return is_filer


def data_fiter(dataset):
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    errors = []
    problem_N = 0
    rN = 0
    for i, input_data in enumerate(data_loader):
        input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, cut_ratio, target_node_typ = input_data
        env = Env(sheet[0, :n_typ_sheet].numpy(), input_seq[0, :n_typ_item].numpy(), edge[0])

        target_seq = target_seq[0, :target_len]
        group_seq = []
        final_seq = []
        for j in range(1, target_len + 1):
            node_typ = target_node_typ[0, j, 0].item()
            # node= [node_typ, bin_id, cut_typ0, cut_typ_out, cut_typ_in, item_id, is_rotated]
            if (node_typ == 1 or node_typ == 2) and len(group_seq) >= 2:  # 准备新开始一个bin 或者 group, 则启动把当前group排进去
                final_seq.extend(env.step_group(group_seq))
                group_seq = []
                if node_typ == 1:
                    if env.new_bin(target_seq[j - 1, 0].item()) == False:
                        print('error matial')
                        exit(1)

                    node = [node_typ, target_seq[j - 1, 0].item(), 0, 0, 0, 0, 0]
                else:
                    node = [node_typ, 0, target_seq[j - 1, 1].item(), target_seq[j - 1, 2].item(),
                            target_seq[j - 1, 3].item(), 0, 0]

            elif node_typ == 1:
                group_seq = []
                node = [node_typ, target_seq[j - 1, 0].item(), 0, 0, 0, 0, 0]
                if env.new_bin(target_seq[j - 1, 0].item()) == False:
                    print('error matial')
                    exit(1)


            elif node_typ == 2:
                node = [node_typ, 0, target_seq[j - 1, 1].item(), target_seq[j - 1, 2].item(),
                        target_seq[j - 1, 3].item(), 0, 0]

            else:  # node_typ == 3
                node = [node_typ, 0, 0, 0, 0, target_seq[j - 1, -2].item(), target_seq[j - 1, -1].item()]

            group_seq.append(node)

        if len(group_seq) > 2:
            env.step_group(group_seq)
        is_filter, rn = env.layout_filter(i, dataset)
        rN += rn
        if is_filter:
            # print('problem data', dataset.data_idx[i])
            problem_N += 1
            errors.append(dataset.data_idx[i])

    with open('problem_cases_merge.txt', 'w', encoding='utf-8') as f:
        for k in errors:
            f.write(k + '\n')
    print('problem: ', problem_N)
    lab_item_seq = []
    total_order_list = []
    total_bins_list = []
    total_edge_list = []
    cut_ratio = []
    for k, v in enumerate(dataset.total_order_list):
        if sum(v[:, -1]) > 0:
            lab_item_seq.append(dataset.lab_item_seq[k])
            total_order_list.append(dataset.total_order_list[k])
            total_bins_list.append(dataset.total_bins_list[k])
            total_edge_list.append(dataset.total_edge_list[k])
            cut_ratio.append(dataset.cut_ratio[k])
    dataset.lab_item_seq = copy.deepcopy(lab_item_seq)
    dataset.total_order_list = copy.deepcopy(total_order_list)
    dataset.total_bins_list = copy.deepcopy(total_bins_list)
    dataset.total_edge_list = copy.deepcopy(total_edge_list)
    dataset.cut_ratio = copy.deepcopy(cut_ratio)
    print('remove num bins: %d' % (rN))
    return dataset


'''

from bbp_dataset import lab_dataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    order_max_len = 200
    order_min_len = 5
    sheet_max_len = 4
    max_target_len = 450
    root = '../data/merge_data'
    data = '../data/test_train.txt'
    dataset = lab_dataset(root, data, False, order_max_len, sheet_max_len, order_min_len, max_target_len)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    # print(len(data_loader))
    errors = []
    train_resort = []
    for i, input_data in enumerate(data_loader):
        input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, cut_ratio, target_node_typ = input_data
        env = Env(sheet[0, :n_typ_sheet].numpy(), input_seq[0, :n_typ_item].numpy(), edge[0])

        target_seq = target_seq[0, :target_len]
        print(target_seq)
        group_seq = []
        final_seq = []
        for j in range(1, target_len + 1):
            node_typ = target_node_typ[0, j, 0].item()
            # node= [node_typ, bin_id, cut_typ0, cut_typ_out, cut_typ_in, item_id, is_rotated]
            if (node_typ == 1 or node_typ == 2) and len(group_seq) >= 2:  # 准备新开始一个bin 或者 group, 则启动把当前group排进去
                final_seq.extend(env.step_group(group_seq))
                group_seq = []
                if node_typ == 1:
                    if env.new_bin(target_seq[j - 1, 0].item()) == False:
                        print('error matial')
                        exit(1)
                    print(len(env.layouts), 'xxxxx')
                    node = [node_typ, target_seq[j - 1, 0].item(), 0, 0, 0, 0, 0]
                else:
                    node = [node_typ, 0, target_seq[j - 1, 1].item(), target_seq[j - 1, 2].item(),
                            target_seq[j - 1, 3].item(), 0, 0]

            elif node_typ == 1:
                group_seq = []
                node = [node_typ, target_seq[j - 1, 0].item(), 0, 0, 0, 0, 0]
                if env.new_bin(target_seq[j - 1, 0].item()) == False:
                    print('error matial')
                    exit(1)
                print(len(env.layouts), 'xxxxx')

            elif node_typ == 2:
                node = [node_typ, 0, target_seq[j - 1, 1].item(), target_seq[j - 1, 2].item(),
                        target_seq[j - 1, 3].item(), 0, 0]

            else:  # node_typ == 3
                node = [node_typ, 0, 0, 0, 0, target_seq[j - 1, -2].item(), target_seq[j - 1, -1].item()]

            group_seq.append(node)

        if len(group_seq) > 2:
            env.step_group(group_seq)

        env.display(dataset.data_idx[i])
        #exit(1)
        #env.layout_filter(i, dataset)
'''
