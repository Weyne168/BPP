import numpy as np
from model.generator_hry import LayoutGenerator
import torch
from typing import List, Optional
from model.bppEnv import Env
import copy
from deepdiff import DeepDiff


class mctNode:
    def __init__(self, state: Optional["Env"] = None, parent: Optional["mctNode"] = None,
                 is_root: Optional["bool"] = False) -> None:
        """
        Initialize a Node.

        :param state: The state or token IDs associated with this Node.
        :param parent: The parent Node of this Node; None if this Node is the root.
        """
        self.state: Env = state
        self.parent: Optional["mctNode"] = parent
        self.children: List["mctNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.is_bin_root = is_root
        self.rank = 1

    def __eq__(self, other):
        if isinstance(other, mctNode):
            seq1 = self.get_data(self.state)
            seq2 = self.get_data(other.state)
            diff = DeepDiff(seq1, seq2)
            if len(diff) > 0:
                return False
            return True
        return False

    def child_not_exist(self, child):
        for c in self.children:
            if c == child:
                return False
        return True

    def get_data(self, env):
        bins = []
        for b in range(0, len(env.layouts)):
            layout = env.layouts[b]
            seq = layout.bin_tree.cut_seq_label()
            bins.extend(seq)
        return bins


class MCTS:
    def __init__(self,
                 model: LayoutGenerator,
                 max_depth: int = 10,
                 num_simulations: int = 10,
                 num_extend: int = 3,
                 gpu_id: int = 0
                 ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.num_extend = num_extend

    def init_env(self, data):
        input_seq, sheet, target_seq, n_typ_item, n_typ_sheet, target_len, edge, cut_ratio, mask_records, _ = data
        '''        
        print(target_seq[0, :target_len[0]].long())
        p = target_seq[0, :target_len[0]].long()
        with open('target_seq.txt', 'w', encoding='utf-8') as f:
            s = ''
            for i in range(p.shape[0]):
                s += str(i) + ':  '
                for j in range(p.shape[1]):
                    s += str(p[i, j].item()) + ', '
                s += '\n'
            f.write(s)

        with open('mk_rds.txt', 'w', encoding='utf-8') as f:
            s = ''
            for i in range(target_len[0]):
                ms = mask_records[0, i, :n_typ_item[0] + 1]
                ss = ms.sum()
                s += str(i) + '=> ' + str(ss.item()) + '== '
                for j in range(n_typ_item[0] + 1):
                    s += str(ms[j].item()) + ', '

                s += '\n'
            f.write(s)
        exit(1)
        '''
        self.n_typ_item = n_typ_item
        self.n_typ_sheet = n_typ_sheet

        batch_size = 1
        pad_mask = torch.zeros(batch_size, self.model.max_item_type + 1)  # add begin node
        bn_mask = torch.zeros(batch_size, self.model.sheet_max)
        pad_mask = pad_mask.to(self.gpu_id, non_blocking=True).long()
        bn_mask = bn_mask.to(self.gpu_id, non_blocking=True).long()

        for b in range(batch_size):
            bn_mask[b, :n_typ_sheet[b]] = sheet[b, :n_typ_sheet[b], 0]
            pad_mask[b, 1:n_typ_item[b] + 1] = input_seq[b, :n_typ_item[b], -1]
        pad_mask[:, 0] = 1

        self.input_seq = input_seq.to(self.gpu_id, non_blocking=True)
        self.sheet = sheet.to(self.gpu_id, non_blocking=True)
        self.edge = edge.to(self.gpu_id, non_blocking=True)

        self.pad_mask = pad_mask
        self.bn_mask = bn_mask

        begin_embedding, bin_size_embedding, item_embedding, item_size_embedding = self.get_init_embedding(sheet,
                                                                                                           bn_mask,
                                                                                                           edge,
                                                                                                           input_seq,
                                                                                                           pad_mask)

        '''
        input_encodding, end_embedding, bin_size_embedding, item_size_embedding = self.model.forward_encode(
            self.input_seq,
            self.sheet,
            self.it_mask,
            self.bn_mask,
            self.edge)
        '''

        self.context_embedding = item_embedding
        self.item_size_embedding = item_size_embedding
        self.bin_size_embedding = bin_size_embedding
        self.begin_embedding = begin_embedding

        np.random.seed(0)
        return cut_ratio

    def search(self, root: mctNode) -> None:
        if root == None:
            env = Env(self.sheet[0, :self.n_typ_sheet].cpu().numpy(),
                      self.input_seq[0, :self.n_typ_item].cpu().numpy(),
                      self.edge[0].item())
            env.new_bin(0)
            root = mctNode(state=env, is_root=True)

        for _ in range(self.num_simulations):
            node = self.selection(root, 0)  # 选择一个无child的node进行扩展
            node = self.expandsion(node, self.num_extend)
            reward, cid = self.simulation(node)
            self.backpropagation(node, reward, cid)
        best_child, best_score = self.get_best_child(root)
        print('best_child:', best_child.state.get_cut_ratio(), best_child.state.get_put_item_num(),
              len(best_child.state.layouts))
        return best_child, best_score

    # 从传递过来的起点,递归选择一个score最大的点
    def selection(self, node: mctNode, depth: int) -> mctNode:
        if not node.children or node.visits == 0 or depth >= 1000:
            return node
        best_child = None
        best_score = -np.inf
        for child in node.children:
            exploration_term = np.sqrt(2 * np.log(node.visits) / child.visits) if child.visits > 0 else 0.0
            score = (child.value / child.visits if child.visits > 0 else 0.0) + exploration_term
            if score > best_score:
                best_score = score
                best_child = child
        return self.selection(best_child, depth + 1)

    def expandsion(self, node: mctNode, expand_num: int):
        env = node.state
        if env.item_num <= env.get_put_item_num():
            return node
        target_seq, target_len, it_records, rx, ry, last_tree_leaf_nodes = self.model.get_data(node.state)
        target_seq = target_seq.to(self.item_size_embedding.device)
        with torch.no_grad():
            input_embedding = self.model.forward_encode(self.context_embedding, self.pad_mask, it_records[:, 0],
                                                        self.begin_embedding)

            target_decodes = self.model.forward_decode(target_seq, self.item_size_embedding, input_embedding,
                                                       self.bin_size_embedding, self.pad_mask, it_records)

            xs, ys, cs = self.model.forward_pos(target_decodes, expand_num, [[rx, ry]])
            lcs, rcs, lrs, rrs = self.model.forward_items(target_decodes, expand_num, it_records[:, target_len - 1])
            nds = self.model.forward_bins(env, target_decodes, expand_num)

        xs = xs[0]
        ys = ys[0]
        cs = cs[0]
        infos = []
        items = env.items[:, :-1]
        for i in range(len(xs)):
            nd_typ = nds[i]
            if nd_typ == 1:
                infos.append((1, 0, 0, 0, 0, 0, 0, 0))
                continue
            elif nd_typ >= 2:
                if env.layouts[-1].bin_tree.get_item_num() > 0:
                    infos.append((nd_typ, 0, 0, 0, 0, 0, 0, 0))
                continue

            x = xs[i]
            y = ys[i]
            c = cs[i]

            if lcs[i] >= items.shape[0]:
                lcs[i] = 1
            lc_w, lc_h = items[lcs[i]]
            lr = lrs[i]
            if lr == 2:
                t = lc_w
                lc_w = lc_h
                lc_h = t

            if rcs[i] >= items.shape[0]:
                rcs[i] = 1
            rc_w, rc_h = items[rcs[i]]
            rr = rrs[i]
            if rr == 2:
                t = rc_w
                rc_w = rc_h
                rc_h = t

            x_fit = float('inf')
            y_fit = float('inf')
            lc = 0
            rc = 0
            lr = 0
            rr = 0
            tx = 5000
            ty = 5000
            for nd in last_tree_leaf_nodes:
                nx, ny = nd.x, nd.y
                W, H = nd.W, nd.H
                nx2, ny2 = nx + W, ny + H
                if x >= nx and y >= ny and x <= nx2 and y <= ny2:
                    if c == 1:  # 横切
                        if y >= ny + node.state.min_unit and y >= ny + lc_h and W >= lc_w and W >= rc_w and ny2 - y >= rc_h and abs(
                                x - nx) < x_fit:
                            tx = nx
                            ty = y.item()
                            x_fit = abs(x - nx)
                            lc = lcs[i]
                            rc = rcs[i]
                            lr = lrs[i]
                            rr = rrs[i]
                        elif y >= ny + node.state.min_unit and y >= ny + lc_h and W >= lc_w and abs(
                                x - nx) < x_fit:
                            tx = nx
                            ty = y.item()
                            x_fit = abs(x - nx)
                            lc = lcs[i]
                            rc = 0
                            lr = lrs[i]
                            rr = 0
                        elif y >= ny + node.state.min_unit and abs(
                                x - nx) < x_fit:
                            tx = nx
                            ty = y.item()
                            x_fit = abs(x - nx)
                            lc = 0
                            rc = 0
                            lr = 0
                            rr = 0

                    else:  # 竖切
                        if x >= nx + node.state.min_unit and x >= nx + lc_w and H >= lc_h and H >= rc_h and nx2 - x >= rc_w and abs(
                                y - ny) < y_fit:
                            tx = x.item()
                            ty = ny
                            y_fit = abs(y - ny)
                            lc = lcs[i]
                            rc = rcs[i]
                            lr = lrs[i]
                            rr = rrs[i]
                        elif x >= nx + node.state.min_unit and x >= nx + lc_w and H >= lc_h and abs(
                                y - ny) < y_fit:
                            tx = x.item()
                            ty = ny
                            y_fit = abs(y - ny)
                            lc = lcs[i]
                            rc = 0
                            lr = lrs[i]
                            rr = 0

                        elif x >= nx + node.state.min_unit and abs(
                                y - ny) < y_fit:
                            tx = x.item()
                            ty = ny
                            y_fit = abs(y - ny)
                            lc = 0
                            rc = 0
                            lr = 0
                            rr = 0

            if tx < 5000 and ty < 5000:
                infos.append((0, c, tx, ty, lc, lr, rc, rr))

        rank = 1
        for s in infos:
            bid, cty, x, y, lc, lr, rc, rr = s
            c_env = copy.deepcopy(node.state)
            item_mask = copy.deepcopy(it_records[0, target_len - 1])
            if bid != 0:
                c_env.layouts[-1].bin_tree.travel_put(item_mask, c_env.items)

                if bid == 1:
                    target_seq, target_len, it_records, rx, ry, last_tree_leaf_nodes = self.model.get_data(node.state,
                                                                                                           True)
                    target_seq = target_seq.to(self.item_size_embedding.device)
                    input_embedding = self.model.forward_encode(self.context_embedding, self.pad_mask, it_records[:, 0],
                                                                self.begin_embedding)
                    target_decodes = self.model.forward_decode(target_seq, self.item_size_embedding, input_embedding,
                                                               self.bin_size_embedding, self.pad_mask, it_records)
                    probs = self.model.cls_node_typ(target_decodes[:, -1]) / 0.3
                    probs[:, 2 + env.sheets.shape[0] - 1:] = float('-inf')
                    probs[:, :2] = float('-inf')
                    probs = torch.softmax(probs, dim=1)[0]
                    bid = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())

                c_env.new_bin(bid - 2)
                child = mctNode(state=c_env, parent=node)  # 只切不放item在子节点
                if node.child_not_exist(child):
                    child.rank = rank
                    node.children.append(child)
                    rank += 1
                continue

            nid = c_env.layouts[-1].bin_tree.create_nodes((x, y), cty, c_env.min_unit, c_env.items[1:], item_mask[1:])
            if nid == -1:
                continue

            if lc > 0:
                left_item = [lc, items[lc, 0], items[lc, 1], lr]
                c_env.layouts[-1].bin_tree.node_list[-2].item = left_item

                if rc > 0:
                    right_item = [rc, items[rc, 0], items[rc, 1], rr]
                    c_env.layouts[-1].bin_tree.node_list[-1].item = right_item

            child = mctNode(state=c_env, parent=node)  # 只切不放item在子节点
            if node.child_not_exist(child):
                child.rank = rank
                node.children.append(child)
                rank += 1

        # 模型扩展失效,规则扩展
        if len(node.children) == 0:
            item_mask = copy.deepcopy(it_records[0, target_len - 1])
            for nd in last_tree_leaf_nodes:
                x0, y0, W, H = nd.x, nd.y, nd.W, nd.H
                p = 0
                S = W * H
                candidate = 0
                for j in range(1, items.shape[0]):
                    if item_mask[j] > 0:
                        w, h = items[j, :2]
                        if (w <= W and h <= H) or (h <= W and w <= H):
                            if w * h / S > p:
                                candidate = j
                                p = w * h / S

                if candidate > 0:
                    w, h = items[candidate, :2]
                    if (w <= W and h <= H):  # item 不旋转
                        # 先竖切再横切
                        item = [candidate, w, h, 1]
                        e = copy.deepcopy(node.state)
                        nid = e.layouts[-1].bin_tree.create_nodes((x0 + w, y0), 2, e.min_unit, e.items[1:],
                                                                  item_mask[1:])
                        if nid != -1:
                            nd_left = e.layouts[-1].bin_tree.node_list[-2]
                            x1, y1 = nd_left.x, nd_left.y
                            nid = e.layouts[-1].bin_tree.create_nodes((x1, y1 + h), 1, e.min_unit, e.items[1:],
                                                                      item_mask[1:])
                            if nid != -1:
                                e.layouts[-1].bin_tree.node_list[-2].item = item
                                child = mctNode(state=e, parent=node)  # 只放左边
                                if node.child_not_exist(child):
                                    child.rank = rank
                                    node.children.append(child)

                        # 先横切再竖切
                        e = copy.deepcopy(node.state)
                        nid = e.layouts[-1].bin_tree.create_nodes((x0, y0 + h), 1, e.min_unit, e.items[1:],
                                                                  item_mask[1:])
                        if nid != -1:
                            nd_left = e.layouts[-1].bin_tree.node_list[-2]
                            x1, y1 = nd_left.x, nd_left.y
                            nid = e.layouts[-1].bin_tree.create_nodes((x1 + w, y1), 2, e.min_unit, e.items[1:],
                                                                      item_mask[1:])
                            if nid != -1:
                                e.layouts[-1].bin_tree.node_list[-2].item = item
                                child = mctNode(state=e, parent=node)  # 只放左边
                                if node.child_not_exist(child):
                                    node.children.append(child)

                    if (h <= W and w <= H):  # item 旋转
                        # 先竖切再横切
                        item = [candidate, w, h, 2]
                        e = copy.deepcopy(node.state)
                        nid = e.layouts[-1].bin_tree.create_nodes((x0 + h, y0), 2, e.min_unit, e.items[1:],
                                                                  item_mask[1:])
                        if nid != -1:
                            nd_left = e.layouts[-1].bin_tree.node_list[-2]
                            x1, y1 = nd_left.x, nd_left.y
                            nid = e.layouts[-1].bin_tree.create_nodes((x1, y1 + w), 1, e.min_unit, e.items[1:],
                                                                      item_mask[1:])
                            if nid != -1:
                                e.layouts[-1].bin_tree.node_list[-2].item = item
                                child = mctNode(state=e, parent=node)  # 只放左边
                                if node.child_not_exist(child):
                                    child.rank = rank
                                    node.children.append(child)

                        # 先横切再竖切
                        e = copy.deepcopy(node.state)
                        nid = e.layouts[-1].bin_tree.create_nodes((x0, y0 + w), 1, e.min_unit, e.items[1:],
                                                                  item_mask[1:])
                        if nid != -1:
                            nd_left = e.layouts[-1].bin_tree.node_list[-2]
                            x1, y1 = nd_left.x, nd_left.y
                            nid = e.layouts[-1].bin_tree.create_nodes((x1 + h, y1), 2, e.min_unit, e.items[1:],
                                                                      item_mask[1:])
                            if nid != -1:
                                e.layouts[-1].bin_tree.node_list[-2].item = item
                                child = mctNode(state=e, parent=node)  # 只放左边
                                if node.child_not_exist(child):
                                    child.rank = rank
                                    node.children.append(child)

        if len(node.children) == 0:
            e = copy.deepcopy(node.state)
            item_mask = copy.deepcopy(it_records[0, target_len - 1])
            _ = e.layouts[-1].bin_tree.travel_put(item_mask, e.items)
            e.new_bin(0)
            child = mctNode(state=e, parent=node)  # 只放左边
            if node.child_not_exist(child):
                rank += 1
                child.rank = rank
                node.children.append(child)
        return node

    def simulation(self, node: mctNode):
        if not node.children:
            return 0.0, -1
        probs = torch.zeros(len(node.children))
        for i in range(len(node.children)):
            probs[i] = 1.0 / np.exp(node.children[i].rank)
        probs = torch.softmax(probs, dim=0)
        cid = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())
        # child = np.random.choice(node.children)
        child = node.children[cid]
        reward = self.evaluate(child)
        print('reward= %.3f' % reward)
        return reward, cid

    def cal_reward(self, cost_chain):
        rs = 1 - cost_chain.cpu().detach().numpy()
        r = rs[-1]
        g = 0.99
        for i in range(len(rs) - 2, -1, -1):
            r += rs[i] * g
            g *= 0.99
        return r / len(rs)

    def evaluate(self, node):
        env = copy.deepcopy(node.state)
        self.model.eval()
        env, cost_chain = self.model.forward_once(env, self.item_size_embedding, self.context_embeding, self.pad_mask,
                                                  self.bin_size_embedding, self.begin_embedding, self.max_depth)

        # target_seq, target_len, it_records, rx, ry, last_tree_leaf_nodes = self.model.get_data(env)
        if env.get_put_item_num() >= env.item_num:
            env.remove_blank_layouts()
        f = env.get_put_item_num() / env.item_num
        g = 0
        for i in range(len(env.layouts)):
            res = env.layouts[i].bin_tree.get_res()
            if res > g:
                g = res

        cost_obs = np.log(len(env.layouts))
        cost_pre = cost_chain[0].item()
        reward = 0.5 * env.get_cut_ratio() + 0.5 * (1 - ((cost_obs + cost_pre) / np.log(201))) + f * g
        reward = (reward if reward > 0 else 0)
        # print(env.get_cut_ratio(), env.get_put_item_num(), env.item_num, cost_pre, cost_obs)
        # else:
        # reward = env.get_cut_ratio()
        # reward = (env.get_cut_ratio() + (env.get_put_item_num() / (1 + env.item_num))) / 2
        # reward = env.get_cut_ratio() * cost_chain
        return reward

    def backpropagation(self, node: mctNode, reward: float, cid: int) -> None:
        """
        Backpropagate the simulation's reward value up the tree.

        :param node: The node to start backpropagation from.
        :param reward: The reward value to propagate.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def get_best_child(self, node: mctNode) -> mctNode:
        """
        Return the best child Node based on average value.

        :param node: The node to get the best child from.
        :return: The best child node.
        """
        best_child = None
        best_score = -np.inf
        for child in node.children:
            score = 0 if child.visits == 0 else child.value / child.visits
            if score > best_score:
                best_child = child
                best_score = score
        return best_child, best_score


mse_loss = torch.nn.MSELoss()
if __name__ == "__main__":
    a = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    b = [(2, 2), (1, 2)]

    print(a)
    print(a.shape)
    a = a.reshape(1, -1)
    print(a)
    exit(1)
    diff = DeepDiff(a, b)  # 自动处理循环引用
    print(len(diff))
    exit(1)

    a = torch.ones(1, 10)
    A = torch.rand(10, 10)

    s = A - a
    print(s)
    s = s ** 2
    print(s)
    s = torch.sum(s, dim=1)
    print(s)
    v, k = s.topk(3)
    print(v)
    print(k)

    p = [(1, 2), (2, 1), (3, 3), (3, 3), (2, 1)]
    p = set(p)
    print(p)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(tensor)
    # 创建一个索引数组，指定每行中要选择的列索引
    indices = torch.tensor([[1], [0], [2]])  # 例如，从第一行选择索引1的元素，从第二行选择索引0的元素，从第三行选择索引2的元素
    print(indices)
    # 使用index_select方法选择元素
    selected_elements = tensor[indices]

    print(selected_elements)
