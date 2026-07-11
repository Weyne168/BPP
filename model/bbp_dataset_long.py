import copy
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random
from model.binTrees import ExpBinTree
from model.utils import *

random.seed(10)


# torch.set_printoptions(precision=1)
def split_train_val(pth, dest, np=0.3):
    train_list = []
    test_list = []
    for root, _, files in os.walk(pth):
        N = len(files)
        p = 0.5
        if np > 0 and np < 1:
            N_test = int(N * np)
            p = np
        else:
            N_test = np

        for file in sorted(files):
            while N_test > 0 and random.random() < p:
                test_list.append(file)
                N_test -= 1
                continue
            train_list.append(file)
    save_txt(test_list, os.path.join(dest, 'test.txt'))
    save_txt(train_list, os.path.join(dest, 'train_G.txt'))


def save_txt(dat, pth):
    with open(pth, 'w') as file:
        for line in dat:
            file.write(line + '\n')


def load_list(pth):
    dat_list = []
    with open(pth, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dat_list.append(line.strip())
    return dat_list


class lab_dataset(Dataset):
    def __init__(self, root, dataset, is_train, max_order=500, sheet_max_len=5, min_order=3, max_seq=1100):
        self.order_max = max_order
        self.sheet_max = sheet_max_len
        self.order_min = min_order
        self.max_input_len = 0
        dat_list = load_list(dataset)
        self._read_data(root, dat_list)
        self.resort2()

        self._clip(max_seq)
        self.max_target_len = max_seq
        self.is_train = is_train

    def __len__(self):
        return len(self.total_order_list)

    def _clip(self, max_seq):
        lab_seq = []
        sheets = []
        edges = []
        orders = []
        cut_ratios = []
        N = len(self.lab_item_seq)

        for i in range(N):
            if len(self.lab_item_seq[i]) < max_seq:
                lab_seq.append(self.lab_item_seq[i])
                sheets.append(self.total_bins_list[i])
                edges.append(self.total_edge_list[i])
                orders.append(self.total_order_list[i])
                if self.max_input_len < len(self.total_order_list[i]):
                    self.max_input_len = len(self.total_order_list[i])
                cut_ratios.append(self.cut_ratio[i])
        self.lab_item_seq = lab_seq
        self.total_bins_list = sheets
        self.total_edge_list = edges
        self.total_order_list = orders
        self.cut_ratio = cut_ratios
        print('total samples %d' % len(self.lab_item_seq))

    def __getitem__(self, idx):
        sheet = self.total_bins_list[idx]  # (n, w,h) 按面积从大到小排列
        edge = self.total_edge_list[idx]
        # final_cut_ratio = self.cut_ratio[idx]
        _lab_seq = self.lab_item_seq[idx]  # 一个订单
        num_bin_typ = sheet.shape[0]

        input_seq = np.zeros((self.order_max, 3), dtype=np.int32)
        sheet_dat = np.zeros((self.sheet_max, 3), dtype=np.int32)
        local_cost = np.zeros((self.max_target_len + 1, 1))
        item_mask = np.zeros((self.max_target_len, self.order_max + 1), dtype=np.int32)
        target_seq = torch.zeros((self.max_target_len, 8))

        sheet_dat[:num_bin_typ] = sheet[...]

        B = len(_lab_seq)  # 原片数量
        p = np.random.choice(range(0, B))
        if p > 0:
            used_lab_seq = _lab_seq[p:]
            remove_seq = _lab_seq[:p]
        else:
            used_lab_seq = _lab_seq
            remove_seq = None

        used_items = copy.deepcopy(self.total_order_list[idx])
        if remove_seq != None:
            for b in range(len(remove_seq)):
                bin = remove_seq[b][0]  # 1块原片
                for k in range(len(bin)):
                    if bin[k][4] in used_items.keys():
                        used_items[bin[k][4]] -= 1
                    if bin[k][6] in used_items.keys():
                        used_items[bin[k][6]] -= 1

        _input_seq = []
        num_item_typ = 0
        item_ids = []

        for k, v in used_items.items():
            if v > 0:
                iw, ih = k.split('_')
                item_ids.append(k)
                _input_seq.append([int(iw), int(ih), v])
                num_item_typ += 1

        _input_seq = np.array(_input_seq)
        tids = np.argsort(-1 * _input_seq[:, 0] * _input_seq[:, 1])
        _input_seq = _input_seq[tids]
        input_seq[:num_item_typ] = _input_seq[...]
        used_area = np.sum(_input_seq[:, 0] * _input_seq[:, 1] * _input_seq[:, 2])
        item_id_map = {'0': 0}
        for it_id in range(0, len(tids)):
            item_id_map[item_ids[tids[it_id]]] = it_id + 1

        B = len(used_lab_seq)
        i, b = 0, 0
        cost_area = 0
        while i < target_seq.shape[0] and b < B:
            if i != 0:
                target_seq[i, 0] = 1  # 上一块结束标识
                local_cost[i] = b + 1
                item_mask[i, 1:num_item_typ + 1] = _input_seq[:num_item_typ, -1]
                i += 1

            bin = used_lab_seq[b][0]  # 1块原片
            _, bw, bh = sheet[bin[0][0] - 2]
            cost_area += bw * bh
            for k in range(len(bin)):
                # node_typ, cx, cy
                target_seq[i, 0] = bin[k][0]
                target_seq[i, 1] = bin[k][1]
                target_seq[i, 2] = bin[k][2]
                target_seq[i, 3] = bin[k][3]
                target_seq[i, 4] = item_id_map[bin[k][4]]
                target_seq[i, 5] = bin[k][5]
                target_seq[i, 6] = item_id_map[bin[k][6]]
                target_seq[i, 7] = bin[k][7]

                item_mask[i, 1:num_item_typ + 1] = _input_seq[:num_item_typ, -1]
                if target_seq[i, 4] > 0:
                    _input_seq[int(target_seq[i, 4]) - 1, -1] -= 1
                if target_seq[i, 6] > 0:
                    _input_seq[int(target_seq[i, 6]) - 1, -1] -= 1
                local_cost[i] = b + 1
                i += 1
            b += 1
        target_seq[i, 0] = 1  # 0是非根节点,1是块结束节点
        local_cost[i] = B
        final_cut_ratio = used_area / cost_area
        return torch.from_numpy(input_seq).long(), torch.from_numpy(
            sheet_dat).long(), target_seq, num_item_typ, num_bin_typ, i, edge, final_cut_ratio, torch.from_numpy(
            item_mask), torch.from_numpy(local_cost).float()

    def _read_data(self, root, dat_list):
        self.total_order_list = []
        self.total_node_list = []
        self.total_bins_list = []
        self.total_edge_list = []
        self.total_cut_routes = []
        self.bin_cut_ratio = []
        self.cut_ratio = []
        self.data_idx = []
        max_num_item_type = 0
        max_W = 0
        max_H = 0
        max_w = 0

        min_w = np.inf
        min_h = np.inf
        # for root, _, files in os.walk(dataset):
        # for file in sorted(dat_list):
        for file in dat_list:
            '''
            if file != '1596334140142260224.txt':
                continue
            '''
            err = False
            bin_max_W = 0
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                idx = 0
                bins = []
                bins_k = {}
                item_k = {}
                bin_node_list = []
                # item_list = []
                bin_cut_ratio = []
                edge = 0
                bin_s = 0
                item_s = 0

                if len(data) < 3:
                    # print('bin num is less than 3! %d' % len(data))
                    continue

                for i in range(len(data)):  # bins
                    if err == True:
                        break

                    node_list = []
                    dat = data[i]
                    edge = dat['minCutDistance']
                    bin_cut_ratio.append(dat['ratio'])
                    l, u, r, d = dat['edgingLeft'], dat['edgingTop'], dat['edgingRight'], dat['edgingBottom']
                    W, H = dat['width'], dat['height']
                    if W > max_W:
                        max_W = W
                    if H > max_H:
                        max_H = H

                    if max(W, H) > bin_max_W:
                        bin_max_W = max(W, H)

                    bin_s += W * H
                    W = W - l - r
                    H = H - u - d
                    items = dat['details']
                    # print(items)
                    # sorted_items = sorted(items, key=lambda x: x['put_sequence'])
                    dis_rotate = False
                    ox, oy = 0, 0
                    if W < H:  # 版图以左下顶点为中心顺时针旋转90度,然后坐标系移动到旋转后的新左下点，#新坐标系原点为原坐标系的右下点
                        dis_rotate = True
                        ox = 0
                        oy = -W
                        t = W
                        W = H
                        H = t

                    k = str(int(W)) + '_' + str(int(H))
                    if k not in bins_k.keys():
                        bins_k[k] = len(bins)
                        bins.append([-1, W, H])

                    for j in range(len(items)):
                        item = items[j]
                        ww, hh = item['displayWidth'], item['displayHeight']  # item在版图上显示宽高
                        x, y = item['position']['x'], item['position']['y']
                        if dis_rotate:
                            x1 = x + ww
                            y1 = y
                            # 旋转90
                            nx = y1
                            ny = -x1

                            # 坐标系平移
                            x = nx - ox
                            y = ny - oy

                            t = ww
                            ww = hh
                            hh = t

                        if ww < min_w:
                            min_w = ww
                        if hh < min_h:
                            min_h = hh
                        item_s += ww * hh

                        if ww >= hh:
                            w = ww
                            h = hh
                            is_rotate = 0
                        else:
                            w = hh
                            h = ww
                            is_rotate = 1

                        if w > max_w:
                            max_w = w
                        if 'cutOriention' not in item.keys():
                            print('eee', file)
                            err = True
                            break

                        cut_typ = item['cutOriention']
                        if cut_typ == None:
                            err = True
                            print('err333', file)
                            break
                        # print(item)
                        cut_route = item['cutRoute']
                        if len(cut_route) < 2:
                            err = True
                            print('eee222', file)
                            break
                        # rank= item['ranks']
                        # exit(1)
                        bin_typ = bins_k[k]
                        k1 = str(int(w)) + '_' + str(int(h))
                        if k1 not in item_k.keys():
                            item_k[k1] = 1
                            # item_list.append([w, h, 1])
                        else:
                            # item_list[item_k[k1]][-1] += 1
                            item_k[k1] += 1

                        if cut_typ == 0 or cut_typ == 2:  # 下切,left 2 right
                            cut_typ = 0
                        else:  # 左切, top 2 down
                            cut_typ = 1

                            # exit(1)
                        node_list.append(
                            [k1, int(x), int(y), int(w), int(h), is_rotate, cut_typ, bin_typ,
                             (int(min(cut_route[0]['x'], cut_route[1]['x'])),
                              int(min(cut_route[0]['y'], cut_route[1]['y'])),
                              int(max(cut_route[0]['x'], cut_route[1]['x'])),
                              int(max(cut_route[0]['y'], cut_route[1]['y']))),
                             (int(W), int(H), int(edge))])

                        idx += 1
                    if len(node_list) == 1:
                        node_list[0][8] = (int(ww), 0, int(ww), int(max(cut_route[1]['y'], cut_route[0]['y'])))

                    bin_node_list.append(node_list)
                    # if idx > self.order_max or idx < self.order_min or len(bins) > self.sheet_max:
                    if idx > self.order_max or len(bins) > self.sheet_max:
                        print('over limit', 'item_num= %d' % idx, 'bin_num= %d' % len(bins))
                    continue

                if self.order_max < len(item_k):
                    print('over limit', 'item_num= %d' % len(item_k))
                    continue
                '''
                if max_num_item_type < len(item_list):
                    max_num_item_type = len(item_list)
                '''
                if max_num_item_type < len(item_k):
                    max_num_item_type = len(item_k)

                if err == True:
                    continue

                if bin_max_W >= 5000:
                    continue
                target_rate = np.ceil(item_s / (bin_s / len(data)))
                if target_rate != len(data):
                    continue
                    # pass
                # self.total_order_list.append(np.array(item_list, dtype=np.int32))
                self.total_order_list.append(item_k)
                self.total_node_list.append(bin_node_list)
                self.total_bins_list.append(np.array(bins, dtype=np.int32))
                self.total_edge_list.append(edge)
                self.data_idx.append(file)
                self.bin_cut_ratio.append(bin_cut_ratio)
                self.cut_ratio.append(item_s / bin_s)
                # self.total_cut_routes.append(np.array(cut_route_list, dtype=np.float32))
        print('max_item_type', max_num_item_type, 'max_W|H', max_W, max_H, 'min-w|min-h', min_w, min_h)
        # print(self.total_node_list)
        # exit(1)

    def pack_items_into_strips(self, bid, item_list, bin):
        rect = [0, 0, bin[1], bin[2]]
        tree = ExpBinTree(rect)
        if tree.gen_tree(item_list) == False:
            return False, None, None

        if tree.pack_item_into_node(item_list) == False:
            return False, None, None

        while tree.tune_tree_depth() > 0:
            continue
        '''
        while tree.tune_tree_cut() > 0:
            continue
        '''
        new_bin = tree.cut_seq_label(bid)
        return True, new_bin, tree

    def resort2(self):
        N = len(self.total_node_list)
        # _ct = {3: 0, 1: 0, 2: 1, 0: 1}
        num_bins = 0
        error_cases = {}
        self.lab_item_seq = []
        for n in range(0, N):
            '''
            if n != 38:
                continue
            '''
            bin_node_list = self.total_node_list[n]
            bin_cut_rates = self.bin_cut_ratio[n]

            '''
            # 按面积大小对item的类型进行编号
            items = self.total_order_list[n]
            tids = np.argsort(-1 * items[:, 0] * items[:, 1])
            self.total_order_list[n] = items[tids]
            tid_map = {}
            for j in range(len(tids)):
                tid_map[tids[j]] = j
            '''
            # 按面积大小对bin的类型进行编号
            bins = self.total_bins_list[n]
            bids = np.argsort(-1 * bins[:, 1] * bins[:, 2])
            self.total_bins_list[n] = bins[bids]

            bid_map = {}
            for j in range(len(bids)):
                bid_map[bids[j]] = j

            B = len(bin_node_list)
            bins = []
            for b in range(0, B):
                node_list = bin_node_list[b]

                I = len(node_list)
                bin = []
                for i in range(I):
                    item = node_list[i]
                    _tid = item[0]
                    _bid = item[7]
                    item[7] = bid_map[_bid]
                    bin.append([copy.deepcopy(item)])

                is_success, bin, tree = self.pack_items_into_strips(item[7], bin, self.total_bins_list[n][item[7]])
                if is_success == False:
                    for i in range(I):
                        item = node_list[i]
                        self.total_order_list[n][item[0]] -= 1

                    if self.data_idx[n] in error_cases.keys():
                        error_cases[self.data_idx[n]] += 1
                    else:
                        error_cases[self.data_idx[n]] = 1
                    continue
                num_bins += 1
                bins.append((copy.deepcopy(bin), bin_cut_rates[b], len(bin)))
                # trees.append(copy.deepcopy(tree))
            # 裁切率从大到小, 裁切率一样,则strips数量多的排前面
            # bins = sorted(bins, key=lambda s: s[2], reverse=True)
            bins = sorted(bins, key=lambda s: s[1], reverse=True)
            nb = 1
            s = 0
            tmp = [bins[s]]
            while nb < len(bins):
                if bins[nb] != bins[s]:
                    tmp = sorted(tmp, key=lambda s: s[2], reverse=True)
                    bins[s:nb] = copy.deepcopy(tmp)
                    s = nb
                    tmp = [bins[s]]
                else:
                    tmp.append(bins[nb])
                nb += 1

            '''
            pth = os.path.join('../checkpoints3', self.data_idx[n])
            if os.path.exists(pth):
                remove_files(pth)
            else:
                os.makedirs(pth)
            for t in range(len(trees)):
                trees[t].display(t, pth)
            exit(1)
            '''
            self.lab_item_seq.append(copy.deepcopy(bins))
        with open('error_cases_merge.txt', 'w', encoding='utf-8') as f:
            for k in error_cases.keys():
                f.write(k + '\n')

        nf = 0
        with open('train_merge_fit.txt', 'w', encoding='utf-8') as f:
            for i in range(len(self.data_idx)):
                if self.data_idx[i] in error_cases.keys():
                    continue
                f.write(self.data_idx[i] + '\n')
                nf += 1
        print('legal:%d | illegal:%d' % (nf, len(error_cases)))
        print('total num bins: %d' % (num_bins))


def w_data(data_file='../data/data.json'):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_json = []
    for dat in data:
        label, edge = dat['label']
        dat['edge'] = edge
        dat['label'] = label
        data_json.append(dat)

    with open('../data/data_2.json', 'w', encoding='utf-8') as f:
        json.dump(data_json, f)


from torch.utils.data import DataLoader

if __name__ == "__main__":
    order_max_len = 200
    order_min_len = 5
    sheet_max_len = 4
    max_target_len = 450

    lab_dat = lab_dataset('../data/merge_data', '../data/resort_merge.txt', True, 200, 4, 5, 450)
    data_loader = DataLoader(
        lab_dat,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    print(len(data_loader))
    exit(1)
    for i, (
            input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, cut_ratio,
            target_node_typ) in enumerate(
        data_loader):
        print('mmmmm')
