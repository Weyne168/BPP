# !/usr/bin/env python3
import copy
import sys
from queue import Queue
import threading
from threading import Thread
import time
import os
import torch
import numpy as np

sys.path.append('/root/bbp_v8')
import json
from flask import Flask, request
import argparse
from model.bppEnv import Env
from utils.my_utils import select_device
from collections import OrderedDict
from model.generator_long import LayoutGenerator
from model.MCTS_deploy import MCTS, mctNode

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

app = Flask(__name__)
order_max_len = 200
order_min_len = 5
sheet_max_len = 4
max_target_len = 450
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 12
torch.manual_seed(seed)  # torch的CPU随机性,为CPU设置随机种子
torch.cuda.manual_seed(seed)  # torch的GPU随机性,为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='PyTorch on BPP with RL')
parser.add_argument('--load_path', type=str, default='./checkpoints',
                    help='Path to load model parameters and optimizer state from')
parser.add_argument('--resume', type=bool, default=True, help='Resume from previous checkpoint file')
parser.add_argument('--resume_epoch', type=int, default=30, help='Resume from previous checkpoint file')

parser.add_argument('--batch', type=int, default=1, help='Learning rate')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
# Network
parser.add_argument('--embedding_dim', type=int, default=1024, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout value')
parser.add_argument('--display', type=bool, default=False, help='')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

generator = LayoutGenerator(embedding_dim=args.embedding_dim, hidden_dim=args.hiddens, order_max_len=order_max_len,
                            sheet_max_len=sheet_max_len)

if args.resume:
    pth = os.path.join(args.load_path, 'epoch-{}.pt'.format(args.resume_epoch))
    print(pth)
    if os.path.exists(pth):
        ck = torch.load(pth, map_location=lambda storage, loc: storage)
        # ck = torch.load(pth, map_location='cuda:{}'.format(rank))
    else:
        print('resumed model missing!')
        exit(1)
    torch.set_rng_state(ck['rng_state'])
    '''
    if args.use_gpu:
        torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
    '''
    new_state_dict = OrderedDict()
    pretrained = ck['generator']
    for k, v in generator.state_dict().items():
        if k in pretrained.keys():
            new_state_dict[k] = pretrained[k]
        else:
            new_state_dict[k] = v
    generator.load_state_dict(new_state_dict, strict=True)
    print('finish loading pretrained model!!')

if args.use_gpu:
    device = select_device(args.gpu)
else:
    device = select_device()
    # torch.backends.cudnn.enabled = False
model = generator.to(device)
model.eval()
from model.bbp_dataset_long import lab_dataset
from torch.utils.data import DataLoader


def data_process2(pth, train_format_file):
    dat_set = lab_dataset(pth, train_format_file, True, order_max_len, sheet_max_len, order_min_len, max_target_len)
    dat_loader = DataLoader(
        dat_set,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    input_info = []
    for i, input_data in enumerate(dat_loader):
        input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, final_cut_ratio, item_records, local_cost = input_data
        print(target_seq[0, :target_len[0].item() + 1])
        # print(target_seq[0, :20])
        _input_seq = torch.zeros((input_seq.shape[0], input_seq.shape[1], 7))
        _input_seq[:, :, :3] = input_seq[...]
        _sheet = torch.zeros((sheet.shape[0], sheet.shape[1], 7))
        _sheet[:, :, :3] = sheet[...]
        input_info.append([torch.Tensor([i]), _input_seq, _sheet, n_typ_item, n_typ_sheet, edge, final_cut_ratio])
        print('===============================')
    return input_info


def data_process(json_data):
    bins_k = {}
    item_k = {}
    order_list = []
    sheet_list = []
    edge_list = []
    case_id_list = []

    label_list = []
    n_case = 0
    for dat in json_data:
        if 'label' in dat.keys():
            label_list.append(dat['label'])
        if 'idx' in dat.keys():
            case_id_list.append(dat['idx'])
        else:
            case_id_list.append(str(n_case))
        n_case += 1

        edge_list.append(dat['edge'])
        sheet = []
        for s in dat['sheet']:
            n, _w, _h, l, u, r, d = s
            W = _w - l - r
            H = _h - u - d
            if W < H:
                t = W
                W = H
                H = t
            k = str(W) + '_' + str(H)
            if k not in bins_k.keys():
                bins_k[k] = len(sheet)
                sheet.append([n, W, H, l, u, r, d])
        sorted_bins = sorted(sheet, key=lambda s: -1 * s[1] * s[2])
        order = []
        input_item_num = 0
        for da in dat['order']:
            n, _w, _h, l, u, r, d = da
            w = _w + l + r
            h = _h + u + d

            if w < h:
                t = w
                w = h
                h = t
            k1 = str(int(w)) + '_' + str(int(h))
            if k1 not in item_k.keys():
                item_k[k1] = len(item_k)
                order.append([w, h, n, l, u, r, d])
            else:
                order[item_k[k1]][2] += n
            input_item_num += n
        sorted_items = sorted(order, key=lambda s: -1 * s[0] * s[1])
        order_list.append(np.array(sorted_items))
        sheet_list.append(np.array(sorted_bins))
    N = len(order_list)
    order_dat = np.zeros((N, order_max_len, 7))
    sheet_dat = np.zeros((N, sheet_max_len, 7))
    order_num = np.zeros((N), np.int32)
    sheet_num = np.zeros((N), np.int32)

    for i in range(N):
        print(order_list[i].shape[0], sheet_list[i].shape[0])
        order_dat[i, :order_list[i].shape[0]] = order_list[i][...]
        sheet_dat[i, :sheet_list[i].shape[0]] = sheet_list[i][...]
        order_num[i] = order_list[i].shape[0]
        sheet_num[i] = sheet_list[i].shape[0]
    print('input_item_num: %d' % input_item_num)

    return case_id_list, torch.from_numpy(order_dat).long(), torch.from_numpy(
        sheet_dat).long(), order_num, sheet_num, edge_list, label_list


def save_layout(env, case_idx, items, sheets):
    dat = []
    bin_sum = 0
    item_sum = 0
    out_put_num = 0
    for i in range(len(env.layouts)):
        layout = env.layouts[i]
        width, height = layout.bin_size

        bs = width * height
        bin_sum += bs
        nds = []
        its = 0

        l, u, r, d = sheets[layout.bin_typ - 1, 3:]
        edge = layout.edge
        for j in range(len(layout.bin_tree.node_list)):
            nd = layout.bin_tree.node_list[j]

            if nd.item == None:
                continue
            item_typ, w, h, is_rotated = nd.item
            item_typ = item_typ - 1
            x, y = nd.x, nd.y

            # w, h = items[item_typ, :2]
            if is_rotated == 1:
                ww = w
                hh = h
            else:
                ww = h
                hh = w

            its += ww * hh
            il, iu, ir, id = items[item_typ, 3:]
            nds.append([int(is_rotated), round(float(x), 1), round(float(y), 1), round(float(w), 1),
                        round(float(h), 1),
                        round(float(ww), 1), round(float(hh), 1), int(il), int(iu), int(ir), int(id)])
            out_put_num += 1
        item_sum += its
        t = {'size': [round(float(width), 1), round(float(height), 1), int(l), int(u), int(r), int(d)],
             'edge': int(edge), 'item': nds, 'ratio': round(float(its / bs), 3)}
        dat.append(t)

    dr = {'order': dat, 'cut_rate': round(float(item_sum / bin_sum), 3)}
    with open('./layouts/%s.json' % case_idx, 'w') as f:
        json.dump(dr, f)
    print('out_put_num: %d' % out_put_num)


def run_batch(data):
    model.eval()
    case_idx, items, sheet, n_typ_item, n_typ_sheet, edge, labels = data
    edge = torch.from_numpy(np.array(edge)).long()
    # labels = torch.from_numpy(np.array([labels]))
    env = Env(sheet[0, :n_typ_sheet[0], :3].numpy(), items[0, :n_typ_item[0], :3].numpy(), edge[0])
    env.new_bin(0)

    batch_size = 1
    it_mask = torch.zeros(batch_size, order_max_len + 1)  # add begin node
    bn_mask = torch.zeros(batch_size, sheet_max_len)
    it_mask = it_mask.to(device, non_blocking=True).long()
    bn_mask = bn_mask.to(device, non_blocking=True).long()

    for b in range(batch_size):
        bn_mask[b, :n_typ_sheet[b]] = sheet[b, :n_typ_sheet[b], 0]
        it_mask[b, 1:n_typ_item[b] + 1] = items[b, :n_typ_item[b], 2]
    it_mask[:, 0] = 1

    items = items.to(device, non_blocking=True)
    sheet = sheet.to(device, non_blocking=True)
    edge = edge.to(device, non_blocking=True)

    input_encodding, end_embedding, bin_size_embedding, item_size_embedding = model.forward_encode(
        items[:, :, :3],
        sheet,
        it_mask,
        bn_mask,
        edge)

    env, cost_chain = model.forward_once(env, item_size_embedding, input_encodding,
                                         bin_size_embedding,
                                         end_embedding, np.inf)

    print(env.get_cut_ratio(), labels)
    env.display('case_%d' % case_idx)
    exit(1)


def run_batch2(data):
    case_idx, items, sheet, n_typ_item, n_typ_sheet, edge, labels = data
    item_id_map = {'0_0': 0}
    item_list = items[0, :n_typ_item[0]]
    item_nums = []
    total_ps = 0
    S = sheet[0, 0, 1] * sheet[0, 0, 2]
    for i in range(item_list.shape[0]):
        w, h, v = item_list[i, :3]
        total_ps += w * h * v
        k = str(int(w.item())) + '_' + str(int(h.item()))
        item_id_map[k] = len(item_id_map)
        item_nums.append(v.item())

    item_nums = np.array(item_nums)
    print('min cost:', torch.ceil(total_ps / S).item())
    print('max cut rate:', total_ps / (torch.ceil(total_ps / S) * S))
    model.eval()
    agent = MCTS(model)
    times = 1
    repeat_times = 0
    order_dat = torch.zeros(1, order_max_len, 7).long()
    order_typ = torch.zeros(1, 1).long()
    order_dat[...] = items[...]
    order_typ[0, 0] = n_typ_item[0]
    # g_offset = 0
    keep_offset = 0
    root = None
    cur_item_nums = copy.deepcopy(item_nums)
    global_env = None
    agent.init_env(data)
    target_cut_rate = (total_ps / (torch.ceil(total_ps / S) * S)).item()
    global_cut_rate = (total_ps / (torch.ceil((total_ps + S) / S) * S)).item()

    while np.sum(cur_item_nums) > 0:
        print('left num:', np.sum(cur_item_nums))
        best_child, best_score = agent.search(root)
        cur_env = best_child.state
        while cur_env.item_num > cur_env.get_put_item_num():
            best_child, best_score = agent.search(best_child)
            cur_env = best_child.state

        cur_env.remove_blank_layouts()
        cur_env.layouts.sort(key=lambda s: s.get_cut_ratio(), reverse=False)
        print('cur put', cur_env.get_put_item_num(), len(cur_env.layouts))

        if len(cur_env.layouts) > 1:  # 不大于1必然是最后一块
            i = 0
            total_cur_rate = 0
            cur_layout_len = 0
            # left_item_num = np.zeros(item_nums.shape)
            while len(cur_env.layouts) > i and times > 0 and len(cur_env.layouts) > 1:
                print('----', cur_env.layouts[i].get_cut_ratio())
                # ave_cur_rate = total_cur_rate / cur_layout_len if cur_layout_len > 0 else 0.9
                if (cur_env.layouts[i].get_cut_ratio()) < min(target_cut_rate, global_cut_rate, 0.9):
                    del cur_env.layouts[i]
                else:
                    print('finish', cur_env.layouts[i].get_cut_ratio())
                    total_cur_rate += cur_env.layouts[i].get_cut_ratio()
                    cur_layout_len += 1
                    i += 1

        if global_env != None:
            if keep_offset > 0:
                agent.init_env(data)
                for j in range(len(cur_env.layouts)):
                    for i in range(len(cur_env.layouts[j].bin_tree.node_list)):
                        if cur_env.layouts[j].bin_tree.node_list[i].item != None:
                            tid = cur_env.layouts[j].bin_tree.node_list[i].item[0]
                            w, h, _ = cur_env.items[tid]
                            tid = item_id_map[str(int(w)) + '_' + str(int(h))]
                            cur_env.layouts[j].bin_tree.node_list[i].item[0] = tid
                cur_env.layouts.extend(global_env.layouts[:keep_offset])
            cur_env.layouts.sort(key=lambda s: s.get_cut_ratio(), reverse=True)

            is_eq = True
            for j in range(min(len(cur_env.layouts), len(global_env.layouts))):
                if cur_env.layouts[j].get_cut_ratio() != global_env.layouts[j].get_cut_ratio():
                    is_eq = False
                    repeat_times = 0
                    keep_offset = 0
                    break

            # if g_offset >= len(cur_env.layouts) and is_eq:  # 本次迭代没有改变
            if is_eq:  # 本次迭代没有改变
                times -= 1
                repeat_times += 1
                cur_item_nums = copy.deepcopy(item_nums)

                # for j in range(len(global_env.layouts)):
                keep_offset = min(repeat_times, len(global_env.layouts))
                for j in range(keep_offset):
                    for i in range(1, len(global_env.layouts[j].bin_tree.node_list)):
                        if global_env.layouts[j].bin_tree.node_list[i].item != None:
                            tid = global_env.layouts[j].bin_tree.node_list[i].item[0]
                            cur_item_nums[tid - 1] -= 1

                print('new', np.sum(cur_item_nums))
                j = 0
                order_dat[...] = 0
                for i in range(n_typ_item[0]):
                    if cur_item_nums[i] > 0:
                        order_dat[0, j] = items[0, i]
                        order_dat[0, j, 2] = cur_item_nums[i]
                        j += 1
                order_typ[0, 0] = j
                agent.init_env(data, order_dat, order_typ)
                if keep_offset == len(global_env.layouts):
                    root = None
                else:
                    _env = Env(agent.sheet[0, :agent.n_typ_sheet].cpu().numpy(),
                               agent.input_seq[0, :agent.n_typ_item].cpu().numpy(),
                               agent.edge[0].item())
                    _env.layouts = copy.deepcopy(global_env.layouts[keep_offset:])
                    for i in range(len(_env.layouts)):
                        for j in range(len(_env.layouts[i].bin_tree.node_list)):
                            if _env.layouts[i].bin_tree.node_list[j].item != None:
                                tid = _env.layouts[i].bin_tree.node_list[j].item[0]
                                w, h, _ = _env.items[tid]
                                tid = agent.item_id_map[str(int(w)) + '_' + str(int(h))]
                                _env.layouts[i].bin_tree.node_list[j].item[0] = tid
                    root = mctNode(state=_env, is_root=True)
                continue

            if root == None:  # 表示上一次是从root=None 重新开始的
                agent.init_env(data)
                for j in range(len(cur_env.layouts)):
                    for i in range(len(cur_env.layouts[j].bin_tree.node_list)):
                        print(cur_env.layouts[j].bin_tree.node_list[i].item)
                        if cur_env.layouts[j].bin_tree.node_list[i].item != None:
                            tid = cur_env.layouts[j].bin_tree.node_list[i].item[0]
                            w, h, _ = cur_env.items[tid]
                            tid = item_id_map[str(int(w)) + '_' + str(int(h))]
                            cur_env.layouts[j].bin_tree.node_list[i].item[0] = tid

                if len(cur_env.layouts) > 0:
                    print('put', cur_env.get_put_item_num(), global_env.get_put_item_num())
                    global_env.layouts.extend(copy.deepcopy(cur_env.layouts))
            else:
                global_env = copy.deepcopy(cur_env)
        else:
            global_env = copy.deepcopy(cur_env)

        times += 1
        global_env.layouts.sort(key=lambda s: s.get_cut_ratio(), reverse=True)
        global_cut_rate = global_env.get_cut_ratio()
        root = mctNode(state=global_env, is_root=True)
        g_offset = len(global_env.layouts)
        print('g_offset', g_offset, flush=True)
        for i in range(g_offset):
            print(i, global_env.layouts[i].get_cut_ratio())

        cur_item_nums = copy.deepcopy(item_nums)
        for i in range(len(global_env.layouts)):
            for j in range(len(global_env.layouts[i].bin_tree.node_list)):
                if global_env.layouts[i].bin_tree.node_list[j].item != None:
                    tid = global_env.layouts[i].bin_tree.node_list[j].item[0]
                    if cur_item_nums[tid - 1] <= 0:
                        print('xxxxxxxxxxxxxx', tid - 1)
                    cur_item_nums[tid - 1] -= 1
    global_env.remove_blank_layouts()
    # global_env.layouts[-1].bin_tree.tune_tree_change()
    # global_env.layouts[-1].bin_tree.tune_space()

    save_layout(global_env, case_idx[0], items[0, :n_typ_item[0].cpu()], sheet[0, :n_typ_sheet[0].cpu()])
    global_env.display('test13')
    print('final cut ratio:', global_env.get_cut_ratio())


data_queue = Queue(maxsize=-1)
lock = threading.Lock()


class Optimum(Thread):
    def __init__(self, queue, thread_lock):
        Thread.__init__(self)
        self.Q = queue
        self.lock = thread_lock

    def run(self):
        # print("Start to process video")
        while True:
            # print('QQQQQ', self.Q.qsize())
            time.sleep(15)
            if self.Q.qsize() > 0:
                lock.acquire()
                json_data = self.Q.get()
                lock.release()
                input_data = data_process(json_data)
                run_batch2(input_data)


@app.route("/infer", methods=["POST"])
def infer():
    json_data = request.json
    # print(json_data['label'])#运行算法时报错，故注释
    lock.acquire()
    data_queue.put(json_data)
    lock.release()
    # print(data_queue.qsize())
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    return 'ok'


# 跑测试任务查询队列任务使用
@app.route("/tasksize", methods=["GET"])
def get_todo_count():
    return str(data_queue.qsize())


# 获取结果数据接口
@app.route("/getlayout", methods=["GET"])
def get_layout():
    layout_id = request.args.get("layout_id")
    layouts_dir = os.path.join(os.path.dirname(__file__), 'layouts')
    if layout_id:
        layout_dir = os.path.join(layouts_dir, str(layout_id))
        if os.path.exists(layout_dir):
            filename = os.path.join(layout_dir, 'dat_num.json')
            with open(filename, 'r') as f:
                data = json.load(f)
                return json.dumps(data)
    return json.dumps("获取数据失败!", ensure_ascii=False, indent=4)


@app.route("/getlayout", methods=["GET"])
def get_layout2():
    layout_id = request.args.get("layout_id")
    layout_dir = os.path.join(os.path.dirname(__file__), 'layouts2')
    if layout_id:
        # layout_dir = os.path.join(layouts_dir, str(layout_id))
        filename = os.path.join(layout_dir, str(layout_id) + ".json")
        if os.path.exists(filename):
            # filename = os.path.join(layout_dir, 'dat_num.json')
            with open(filename, 'r') as f:
                data = json.load(f)
                return json.dumps(data)

    return json.dumps("获取数据失败!", ensure_ascii=False, indent=4)


if __name__ == "__main__":
    '''
    opt = Optimum(data_queue, lock)
    opt.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
    opt.join()
    '''
    print(torch.__version__)

    input_data = data_process2('/root/bbp_v8/data/merge_data', '/root/bbp_v8/data/test_train.txt')
    # run_batch(input_data)
    # exit(1)

    data_file = './data/test13.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # print(data[0])

    # input_data = data_process([data[0]])
    # run_batch(input_data)
    run_batch2(input_data[0])
