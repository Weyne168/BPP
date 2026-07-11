# -*- coding: UTF-8 -*-
import copy
import torch.nn.functional
import torch.nn as nn
from torch.nn import Parameter
from model.utils import *
import math
from model.transformer.transformer import *


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, typ='W'):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        if typ == 'W':
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.cos(position * div_term)
            pe[:, 1::2] = torch.sin(position * div_term)

        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        # x==0的时候返回全0向量
        _x = x.clone()
        _x[x < 0] = 0
        res = self.pe[_x]
        _m = x.clone()
        _m[x < 0] = 0
        if mask != None:
            _m[mask != 0] = 1
        else:
            _m[x > 0] = 1
        res = res * _m.unsqueeze(-1)
        return res


class LayoutGenerator(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=768, order_max_len=500,
                 sheet_max_len=5, n_layers=12, attn_heads=8, dropout=0.1):
        """
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        super(LayoutGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.begin_embedding = Parameter(torch.FloatTensor(embedding_dim), requires_grad=True)
        self.embed_WX = PositionalEmbedding(embedding_dim // 2, 5000)
        self.embed_HY2 = PositionalEmbedding(embedding_dim // 2, 3600, 'H')
        # self.embed_item_num = nn.Parameter(torch.ones(1, embedding_dim // 2), requires_grad=True)
        self.embed_item_num = PositionalEmbedding(embedding_dim, 500)

        self.embed_is_rotate = nn.Parameter(torch.ones(2, embedding_dim), requires_grad=True)
        self.none_embedding = Parameter(torch.zeros(1, embedding_dim), requires_grad=False)
        self.end_embedding = Parameter(torch.ones(1, embedding_dim), requires_grad=True)
        self.begin_embedding = Parameter(torch.zeros(1, embedding_dim), requires_grad=True)

        # self.embed_cuts = nn.Linear(3, embedding_dim // 4)
        self.embed_cuts = nn.Parameter(torch.ones(2, embedding_dim), requires_grad=True)

        encoder_layer = TransformerEncoderLayer(n_layers, embedding_dim, attn_heads, hidden_dim, dropout,
                                                batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, n_layers)
        decoder_layer = TransformerDecoderLayer(n_layers, embedding_dim, attn_heads, hidden_dim, dropout,
                                                batch_first=True)

        self.decoder = TransformerDecoder(decoder_layer, n_layers)
        # Transformer的Attention矩阵中引入下三角形形式的Mask，并将输入输出错开一位训练
        self.max_item_type = order_max_len  # 170
        self.sheet_max = sheet_max_len
        self.cls_l_item = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(),
                                        nn.Linear(embedding_dim // 2, self.max_item_type + 1))
        self.cls_r_item = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(),
                                        nn.Linear(embedding_dim // 2, self.max_item_type + 1))
        # none, bin_typ0, bin_typ1,...,bin_typm
        self.cls_node_typ = nn.Linear(embedding_dim, self.sheet_max + 2)
        # none, 不旋转, 旋转
        self.cls_r_rotate = nn.Linear(embedding_dim, 3)
        self.cls_l_rotate = nn.Linear(embedding_dim, 3)

        self.reg_cut_level_1 = nn.Linear(embedding_dim, 125 * 90)  # W:4096*H:4096
        self.reg_cut_level_m = nn.Sequential(nn.Linear(125 * 90, embedding_dim), nn.ReLU())  # W:4096*H:4096
        self.reg_cut_level_2 = nn.Linear(embedding_dim, 40 * 40)
        self.cls_cut_typ = nn.Linear(embedding_dim, 3)
        self.pred_cost = nn.Linear(embedding_dim, 1)
        self.pred_rate = nn.Linear(embedding_dim, 1)
        nn.init.kaiming_normal_(self.embed_is_rotate)
        nn.init.normal_(self.begin_embedding)
        nn.init.normal_(self.end_embedding)

    def ste_round(self, x):
        return torch.round(x) - x.detach() + x

    def get_data(self, env):
        bins = []
        for b in range(0, len(env.layouts)):
            layout = env.layouts[b]
            seq = layout.bin_tree.cut_seq_label()
            bins.extend(seq)

        l1_range = torch.zeros(1, 125, 90)
        l2_range = torch.ones(1, 40, 40)
        last_tree = env.layouts[-1].bin_tree.node_list
        leaf_nodes = []
        for i in range(len(last_tree)):
            nd = last_tree[i]
            if nd.left == -1 and nd.right == -1 and nd.W != 0 and nd.H != 0 and nd.item == None:
                _x1 = nd.x // 40
                _x2 = (nd.x + nd.W) // 40
                _y1 = nd.y // 40
                _y2 = (nd.y + nd.H) // 40
                l1_range[0, _x1:_x2 + 1, _y1:_y2 + 1] = 1
                '''
                _x1 = nd.x % 40
                _x2 = (nd.x + nd.W) % 40
                _y1 = nd.y % 40
                _y2 = (nd.y + nd.H) % 40
                if _x2 > _x1 and _y2 > _y1:
                    l2_range[0, _x1:_x2 + 1, _y1:_y2 + 1] = 1

                elif _x2 < _x1 and _y2 > _y1:
                    l2_range[0, _x1:40, _y1:_y2 + 1] = 1
                    l2_range[0, 0:_x2 + 1, _y1:_y2 + 1] = 1

                elif _x2 > _x1 and _y2 < _y1:
                    l2_range[0, _x1:_x2 + 1, _y1:40] = 1
                    l2_range[0, _x1:_x2 + 1, 0:_y2 + 1] = 1

                elif _x2 < _x1 and _y2 < _y1:
                    l2_range[0, _x1:40, _y1:40] = 1
                    l2_range[0, 0:_x2 + 1, 0:_y2 + 1] = 1

                elif _x2 > _x1 and _y2 == _y1:
                    l2_range[0, _x1:_x2 + 1, :] = 1

                elif _x2 < _x1 and _y2 == _y1:
                    l2_range[0, _x1:40, :] = 1
                    l2_range[0, 0:_x2 + 1, :] = 1

                elif _x2 == _x1 and _y2 > _y1:
                    l2_range[0, :, _y1:_y2 + 1] = 1

                elif _x2 == _x1 and _y2 < _y1:
                    l2_range[0, :, _y1:40] = 1
                    l2_range[0, :, 0:_y2 + 1] = 1

                else:
                    l2_range[0, :, :] = 1
                '''
                leaf_nodes.append(nd)

        l1_range = l1_range.reshape(1, -1)
        l2_range = l2_range.reshape(1, -1)
        item_nums = copy.deepcopy(env.items[1:, -1])
        target_len = len(bins)
        last_tree_offset = 0
        if target_len > 0:
            target_seq = torch.zeros((1, target_len, 8))
            item_mask_seq = np.zeros((1, target_len, self.max_item_type + 1), dtype=np.int32)

            i = 0
            while i < target_len:
                target_seq[0, i, 0] = bins[i][0]
                target_seq[0, i, 1] = bins[i][1]
                target_seq[0, i, 2] = bins[i][2]
                target_seq[0, i, 3] = bins[i][3]
                target_seq[0, i, 4] = bins[i][4]
                target_seq[0, i, 5] = bins[i][5]
                target_seq[0, i, 6] = bins[i][6]
                target_seq[0, i, 7] = bins[i][7]

                item_mask_seq[0, i, 1:item_nums.shape[0] + 1] = item_nums[:item_nums.shape[0]]
                if target_seq[0, i, 4] > 0:
                    item_nums[int(target_seq[0, i, 4]) - 1] -= 1
                if target_seq[0, i, 6] > 0:
                    item_nums[int(target_seq[0, i, 6]) - 1] -= 1
                if bins[i][0] >= 2:
                    last_tree_offset = i
                i += 1
        else:
            target_seq = None
            item_mask_seq = None
        return target_seq, target_len - last_tree_offset, item_mask_seq, l1_range, l2_range, leaf_nodes

    def forward_encode(self, input_seq, sheet, it_mask, bn_mask, edge):
        batch = input_seq.shape[0]
        it_mask_embedding = self.embed_item_num(it_mask[:, 1:].long())

        item_w = input_seq[:, :, 0].clone()
        item_h = input_seq[:, :, 1].clone()

        item_w_embedding = self.embed_WX(item_w.long(), it_mask[:, 1:])
        item_h_embedding = self.embed_HY2(item_h.long(), it_mask[:, 1:])

        item_size_embedding = torch.cat([item_w_embedding, item_h_embedding], dim=-1)
        # item_embedding = torch.cat([item_size_embedding, it_mask_embedding], dim=-1)
        item_embedding = item_size_embedding + it_mask_embedding

        bin_w = sheet[:, :, 1].clone()
        bin_h = sheet[:, :, 2].clone()

        bin_w_embedding = self.embed_WX(bin_w.long(), bn_mask)
        bin_h_embedding = self.embed_HY2(bin_h.long(), bn_mask)
        bin_size_embedding = torch.cat([bin_w_embedding, bin_h_embedding], dim=-1)  # [:, 0]

        edge_embedding = torch.cat([self.embed_WX(edge.long()), self.embed_HY2(edge.long())],
                                   dim=-1).unsqueeze(1)

        end_embedding = self.end_embedding.unsqueeze(0).repeat(batch, 1, 1)
        beg_embedding = self.begin_embedding.unsqueeze(0).repeat(batch, 1, 1)

        begin_embedding = torch.mean(bin_size_embedding, dim=1, keepdim=True) + edge_embedding + beg_embedding
        input_seq_embedding = torch.cat([begin_embedding, item_embedding], dim=1)
        '''
        
        for b in range(batch):
            m = torch.sum(torch.gt(it_mask[b], 0))
            input_seq_embedding[b, m, :] = end_embedding[b]
        '''
        input_encodding = self.encoder(input_seq_embedding, src_key_padding_mask=(it_mask <= 0))
        return input_encodding, end_embedding, bin_size_embedding, item_size_embedding

    def forward_decode(self, target_seq, item_size_embedding, context_embedding, end_embedding, bin_size_embedding,
                       it_mask, it_records):

        if target_seq != None:
            batch = target_seq.shape[0]
        else:
            batch = 1
        #################################  Decoder  ###########################################################
        none_embedding = self.none_embedding.unsqueeze(0).repeat(batch, 1, 1)
        bin_size_embedding = torch.cat([none_embedding, end_embedding, bin_size_embedding], dim=1)
        rotate_embedding = self.embed_is_rotate.unsqueeze(0).repeat(batch, 1, 1)
        rotate_embedding = torch.cat([none_embedding, rotate_embedding], dim=1)
        cut_dir_embedding = self.embed_cuts.unsqueeze(0).repeat(batch, 1, 1)
        cut_dir_embedding = torch.cat([none_embedding, cut_dir_embedding], dim=1)

        item_nodes = torch.cat([none_embedding, item_size_embedding], dim=1)
        target_node_embedding = [context_embedding[:, 0, :].unsqueeze(1)]
        if target_seq != None:
            for b in range(batch):
                bid_embedding = bin_size_embedding[b, target_seq[b, :, 0].long()]
                lc_item = item_nodes[b, target_seq[b, :, 4].long()]
                left_it_nums = torch.zeros(target_seq[b, :, 4].shape).to(item_nodes.device)
                right_it_nums = torch.zeros(target_seq[b, :, 6].shape).to(item_nodes.device)
                for i in range(target_seq.shape[1]):
                    left_it_nums[i] = it_records[b, i, target_seq[b, i, 4].long()]

                lc_rotate = rotate_embedding[b, target_seq[b, :, 5].long()]
                lc_item_num = self.embed_item_num(left_it_nums.long())
                lc_embedding = lc_item + lc_rotate + lc_item_num

                rc_item = item_nodes[b, target_seq[b, :, 6].long()]
                for i in range(target_seq.shape[1]):
                    right_it_nums[i] = it_records[b, i, target_seq[b, i, 6].long()]

                rc_rotate = rotate_embedding[b, target_seq[b, :, 7].long()]
                rc_item_num = self.embed_item_num(right_it_nums.long())
                rc_embedding = rc_item + rc_rotate + rc_item_num

                ct_x = self.embed_WX(target_seq[b, :, 1].long())
                ct_y = self.embed_HY2(target_seq[b, :, 2].long())
                cut_xy = torch.cat([ct_x, ct_y], dim=-1)
                ct = cut_dir_embedding[b, target_seq[b, :, 3].long()] + cut_xy

                tgt_node = bid_embedding + ct + lc_embedding + rc_embedding
                target_node_embedding.append(tgt_node.unsqueeze(0))

        target_node_embedding = torch.cat(target_node_embedding, dim=1)
        sz = target_node_embedding.shape[1]
        tg_mask = torch.zeros(batch, sz)  # add begin
        tgm = tg_mask.to(target_node_embedding.device)
        tgm[0, :sz] = 1
        target_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=target_node_embedding.device), diagonal=1)
        # target_mask = torch.triu(torch.full((sz, sz), -1e9, device=target_node_embedding.device), diagonal=1)
        target_decodding = self.decoder(target_node_embedding, context_embedding, target_mask,
                                        tgt_key_padding_mask=(tgm <= 0), memory_key_padding_mask=(it_mask <= 0))
        return target_decodding

    def forward_pos(self, target_decodes, expand_num, xy_range):
        batch = target_decodes.shape[0]
        with torch.no_grad():
            cut_level_1 = self.reg_cut_level_1(target_decodes[:, -1])
            cut_level_2 = self.reg_cut_level_m(cut_level_1) + target_decodes[:, -1]
        cut_level_1 /= 0.3
        cut_level_2 = self.reg_cut_level_2(cut_level_2) / 0.3
        cut_typ = self.cls_cut_typ(target_decodes[:, -1]) / 0.3
        probs = torch.softmax(cut_typ[:, 1:], dim=1)[0]

        xs = []
        ys = []
        cut_typs = []

        for b in range(batch):
            l1_range, l2_range = xy_range[b]
            cut_level_1[l1_range == 0] = float('-inf')
            level_1_pbs = torch.softmax(cut_level_1, dim=1)[0]
            p1, ct_level_1 = level_1_pbs.topk(expand_num)
            x_l1 = ct_level_1 // 90
            y_l1 = ct_level_1 % 90

            cut_level_2[l2_range == 0] = float('-inf')
            level_2_pbs = torch.softmax(cut_level_2, dim=1)[0]
            p2, ct_level_2 = level_2_pbs.topk(expand_num)
            x_l2 = ct_level_2 // 40
            y_l2 = ct_level_2 % 40

            x = x_l1 * 40 + x_l2
            y = y_l1 * 40 + y_l2

            xs.append(x)
            ys.append(y)

            cs = []
            # pcs = []
            for i in range(expand_num):
                cut_typ = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy()) + 1
                cs.append(cut_typ)
                # pcs.append(probs[cut_typ - 1].item())
            cut_typs.append(cs)
        return xs, ys, cut_typs

    def forward_items(self, target_decodes, expand_num, item_mask):
        with torch.no_grad():
            lr_probs = self.cls_l_rotate(target_decodes[:, -1]) / 0.3
            lr_probs = torch.softmax(lr_probs[:, 1:], dim=1)[0]
            rr_probs = self.cls_r_rotate(target_decodes[:, -1]) / 0.3
            rr_probs = torch.softmax(rr_probs[:, 1:], dim=1)[0]

            lc_probs = self.cls_l_item(target_decodes[:, -1]) / 0.3
            p0 = lc_probs[0, 0].item()
            lc_probs[item_mask <= 0] = float('-inf')
            lc_probs[0, 0] = p0
            lc_probs = torch.softmax(lc_probs, dim=1)[0]
            _, lc = lc_probs.topk(expand_num)
            lc = lc.cpu().numpy()

            rc_probs = self.cls_r_item(target_decodes[:, -1]) / 0.3
            p0 = rc_probs[0, 0].item()
            rc = rc_probs[0, :expand_num].long()
            rc = rc.cpu().numpy()

        lrs, rrs = [], []
        for i in range(len(lc)):
            if lc[i] > 0:
                probs = copy.deepcopy(rc_probs)
                m = copy.deepcopy(item_mask)
                m[0, lc[i]] -= 1
                probs[m <= 0] = float('-inf')
                probs[0, 0] = p0
                probs = torch.softmax(probs, dim=1)[0]
                rc[i] = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())
                if rc[i] > 0:
                    rr = np.random.choice(rr_probs.shape[0], p=rr_probs.cpu().detach().numpy()) + 1
                else:
                    rr = 0

                lr = np.random.choice(lr_probs.shape[0], p=lr_probs.cpu().detach().numpy()) + 1
                lrs.append(lr)
                rrs.append(rr)
            else:
                rc[i] = 0
                lrs.append(0)
                rrs.append(0)

        return lc, rc, lrs, rrs

    def forward_bins(self, env, target_decodes, expand_num):
        node_typs = []
        with torch.no_grad():
            probs = self.cls_node_typ(target_decodes[:, -1]) / 0.3
            probs[:, 2 + env.sheets.shape[0] - 1:] = float('-inf')
            probs = torch.softmax(probs, dim=1)[0]

            while len(node_typs) < expand_num:
                node_typ = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())
                if node_typ < 2 + env.sheets.shape[0] - 1:
                    node_typs.append(node_typ)
        return node_typs

    def forward(self, input_seq, target_seq, sheet, it_mask, bn_mask, tg_mask, edge, it_records):
        batch = input_seq.shape[0]
        input_embedding, end_embedding, bin_size_embedding, item_size_embedding = self.forward_encode(input_seq,
                                                                                                      sheet,
                                                                                                      it_mask,
                                                                                                      bn_mask,
                                                                                                      edge)

        #################################  Decoder  ###########################################################
        none_embedding = self.none_embedding.unsqueeze(0).repeat(batch, 1, 1)
        bin_size_embedding = torch.cat([none_embedding, end_embedding, bin_size_embedding], dim=1)
        rotate_embedding = self.embed_is_rotate.unsqueeze(0).repeat(batch, 1, 1)
        rotate_embedding = torch.cat([none_embedding, rotate_embedding], dim=1)
        cut_dir_embedding = self.embed_cuts.unsqueeze(0).repeat(batch, 1, 1)
        cut_dir_embedding = torch.cat([none_embedding, cut_dir_embedding], dim=1)

        item_nodes = torch.cat([none_embedding, item_size_embedding], dim=1)
        target_node_embedding = []
        level_1_lab = []
        level_2_lab = []
        for b in range(batch):
            bid_embedding = bin_size_embedding[b, target_seq[b, :, 0].long()]

            lc_item = item_nodes[b, target_seq[b, :, 4].long()]
            left_it_nums = torch.zeros(target_seq[b, :, 4].shape).to(item_nodes.device)
            right_it_nums = torch.zeros(target_seq[b, :, 6].shape).to(item_nodes.device)
            for i in range(target_seq.shape[1]):
                left_it_nums[i] = it_records[b, i, target_seq[b, i, 4].long()]

            lc_rotate = rotate_embedding[b, target_seq[b, :, 5].long()]
            lc_item_num = self.embed_item_num(left_it_nums.long())
            lc_embedding = lc_item + lc_rotate + lc_item_num

            rc_item = item_nodes[b, target_seq[b, :, 6].long()]
            for i in range(target_seq.shape[1]):
                right_it_nums[i] = it_records[b, i, target_seq[b, i, 6].long()]

            rc_rotate = rotate_embedding[b, target_seq[b, :, 7].long()]
            rc_item_num = self.embed_item_num(right_it_nums.long())
            rc_embedding = rc_item + rc_rotate + rc_item_num

            ct_x = self.embed_WX(target_seq[b, :, 1].long())
            ct_y = self.embed_HY2(target_seq[b, :, 2].long())
            cut_xy = torch.cat([ct_x, ct_y], dim=-1)
            ct = cut_dir_embedding[b, target_seq[b, :, 3].long()] + cut_xy

            x_level_1 = target_seq[b, :, 1].long() // 40
            y_level_1 = target_seq[b, :, 2].long() // 40
            xy_level_1 = x_level_1 * 90 + y_level_1
            x_level_2 = target_seq[b, :, 1].long() % 40
            y_level_2 = target_seq[b, :, 2].long() % 40
            xy_level_2 = (x_level_2) * 40 + y_level_2
            level_1_lab.append(xy_level_1.unsqueeze(0))
            level_2_lab.append(xy_level_2.unsqueeze(0))

            tgt_node = bid_embedding + ct + lc_embedding + rc_embedding
            target_node_embedding.append(tgt_node.unsqueeze(0))

        target_node_embedding = torch.cat(target_node_embedding, dim=0)
        target_node_embedding = torch.cat([input_embedding[:, 0, :].unsqueeze(1), target_node_embedding], dim=1)
        tgm = tg_mask.to(target_node_embedding.device)
        sz = target_node_embedding.shape[1]
        target_mask = torch.triu(torch.full((sz, sz), float('-inf'), device=target_node_embedding.device), diagonal=1)
        # target_mask = torch.triu(torch.full((sz, sz), -1e9, device=target_node_embedding.device), diagonal=1)
        target_decodding = self.decoder(target_node_embedding, input_embedding[:, 1:], target_mask,
                                        tgt_key_padding_mask=(tgm <= 0), memory_key_padding_mask=(it_mask[:, 1:] <= 0))

        bin_id = self.cls_node_typ(target_decodding)
        # cut_pos = nn.functional.sigmoid(self.reg_cut_pos(target_decodding))
        cut_level_1 = self.reg_cut_level_1(target_decodding)
        cut_level_2 = self.reg_cut_level_m(cut_level_1) + target_decodding
        cut_level_2 = self.reg_cut_level_2(cut_level_2)

        level_1_lab = torch.cat(level_1_lab, dim=0)
        level_2_lab = torch.cat(level_2_lab, dim=0)

        cut_typ = self.cls_cut_typ(target_decodding)  # .unsqueeze(-2)
        lc = self.cls_l_item(target_decodding)
        rc = self.cls_r_item(target_decodding)
        lr = self.cls_l_rotate(target_decodding)
        rr = self.cls_r_rotate(target_decodding)
        # pred_cost_num = nn.functional.sigmoid(self.pred_cost(target_decodding))
        pred_cost_num = nn.functional.relu(self.pred_cost(target_decodding))
        pred_rate = nn.functional.sigmoid(self.pred_rate(target_decodding))
        return bin_id, cut_level_1, cut_level_2, cut_typ, lc, rc, lr, rr, level_1_lab, level_2_lab, pred_cost_num, pred_rate

    def forward_once(self, env, item_size_embedding, context_embedding, bin_size_embedding, end_embedding, max_depth):
        batch = 1
        item_mask = torch.zeros(batch, self.max_item_type + 1)
        it_mask = copy.deepcopy(env.items[:, -1])
        it_mask[0] = 1
        item_mask[0, :it_mask.shape[0]] = torch.from_numpy(it_mask)[...]
        item_mask = item_mask.to(item_size_embedding.device, non_blocking=True).long()

        _, target_len, it_records, _, _, _ = self.get_data(env)
        last_item_records = it_records[0, target_len - 1]
        last_item_records = torch.from_numpy(last_item_records)
        step = 0
        cut_step = 0
        renew = True
        target_decodes = None
        topN = 5
        while torch.sum(last_item_records) > 0 and 500 > step and cut_step < max_depth:
            if renew:
                target_seq, target_len, it_records, r_l1, r_l2, last_tree_leaf_nodes = self.get_data(env)
                # print(target_seq)
                target_seq = target_seq.to(item_size_embedding.device)
                last_item_records = it_records[0, target_len - 1]
                last_item_records = torch.from_numpy(last_item_records)
                last_item_records = last_item_records.to(item_size_embedding.device)
                with torch.no_grad():
                    target_decodes = self.forward_decode(target_seq, item_size_embedding, context_embedding,
                                                         end_embedding,
                                                         bin_size_embedding,
                                                         item_mask, it_records)
                # cost_chain = nn.functional.sigmoid(self.pred_cost(target_decodes[:, -1]))
                # cost_chain.append(cur_pred_cost[0, 0].item())
            # r_l1[...] = 1
            # r_l2[...] = 1
            xs, ys, cut_typs = self.forward_pos(target_decodes, topN, [[r_l1, r_l2]])
            bin_id = self.cls_node_typ(target_decodes[:, -1]) / 0.3
            bin_id[:, 2 + env.sheets.shape[0] - 1:] = float('-inf')
            probs = torch.softmax(bin_id, dim=1)[0]  # 不等于非根节点和结束节点,0非根节点,1 结束节点
            node_typ = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())
            while (node_typ == 1 and torch.sum(last_item_records) > 0) or (
                    node_typ == 0 and len(last_tree_leaf_nodes) == 0):
                bin_id[:, node_typ] = float('-inf')
                probs = torch.softmax(bin_id, dim=1)[0]  # 不等于非根节点和结束节点,0非根节点,1 结束节点
                node_typ = np.random.choice(probs.shape[0], p=probs.cpu().detach().numpy())

            # 当前原片已经没有item为None的叶子节点了,只能新开一片
            if (node_typ != 0 and env.layouts[-1].bin_tree.get_item_num() != 0) or (
                    r_l1.sum() == 0 and r_l2.sum() == 0):
                env.layouts[-1].bin_tree.travel_put(last_item_records, env.items)
                env.new_bin(node_typ - 2)
                step += 1
                cut_step += 1
                renew = True
                continue

            xs = xs[0]
            ys = ys[0]
            cut_typs = cut_typs[0]
            tx, ty = xs[0], ys[0]
            for i in range(len(cut_typs)):
                cut_typ = cut_typs[i]
                if cut_typ == 1:  # 横切
                    y = ys[i]
                    x_fit = float('inf')
                    for nd in last_tree_leaf_nodes:
                        nx, ny = nd.x, nd.y
                        W, H = nd.W, nd.H
                        nx2 = nx + W
                        ny2 = ny + H
                        if y >= ny + env.min_unit and y < ny2 and nx <= tx and tx < nx2 and abs(xs[i] - nx) < x_fit:
                            tx = nx
                            ty = y.item()
                            x_fit = abs(xs[i] - nx)
                    if ty < 50000:
                        break

                else:  # 竖切
                    x = xs[i]
                    y_fit = float('inf')
                    for nd in last_tree_leaf_nodes:
                        nx, ny = nd.x, nd.y
                        W, H = nd.W, nd.H
                        nx2 = nx + W
                        ny2 = ny + H
                        if x >= nx + env.min_unit and x < nx2 and ny <= ty and ty < ny2 and abs(ys[i] - ny) < y_fit:
                            tx = x.item()
                            ty = ny
                            y_fit = abs(ys[i] - ny)

                    if tx < 50000:
                        break

            nid = env.layouts[-1].bin_tree.create_nodes([tx, ty], cut_typ, env.min_unit, env.items[1:],
                                                        last_item_records[1:].cpu().numpy())
            if nid == -1:
                '''
                x_l1 = tx // 40
                y_l1 = ty // 40
                xy_l1 = x_l1 * 90 + y_l1
                r_l1[0, xy_l1] = 0
                x_l2 = tx % 40
                y_l2 = ty % 40
                xy_l2 = x_l2 * 40 + y_l2
                r_l2[0, xy_l2] = 0
                '''
                step += 1
                renew = False
                topN += 1
                continue

            renew = True
            topN = 5
            cut_step += 1
            probs = self.cls_l_item(target_decodes[:, -1]) / 0.3
            p0 = probs[0, 0].item()
            probs[last_item_records.unsqueeze(0) <= 0] = float('-inf')
            probs[0, 0] = p0
            lc_probs = torch.softmax(probs, dim=1)[0]
            lc = np.random.choice(lc_probs.shape[0], p=lc_probs.cpu().detach().numpy())
            try_times = 10
            while lc != 0 and try_times > 0:
                if last_item_records[lc] > 0:
                    left_nid = env.layouts[-1].bin_tree.node_list[nid].left
                    lr_probs = self.cls_l_rotate(target_decodes[:, -1]) / 0.3
                    lr_probs = torch.softmax(lr_probs[:, 1:], dim=1)[0]
                    lr = np.random.choice(lr_probs.shape[0], p=lr_probs.cpu().detach().numpy()) + 1

                    if env.layouts[-1].bin_tree.put_item(left_nid, lc, env.items[lc, :2], lr):
                        last_item_records[lc] -= 1
                        break

                    if lr == 1:
                        lr = 2
                    else:
                        lr = 1

                    if env.layouts[-1].bin_tree.put_item(left_nid, lc, env.items[lc, :2], lr):
                        last_item_records[lc] -= 1
                        break
                    probs[0, lc] = float('-inf')
                lc_probs = torch.softmax(probs, dim=1)[0]
                lc = np.random.choice(lc_probs.shape[0], p=lc_probs.cpu().detach().numpy())
                try_times -= 1
                if try_times == 0:
                    lc = 0

            if lc != 0:
                probs = self.cls_r_item(target_decodes[:, -1]) / 0.3
                p0 = probs[0, 0].item()
                probs[last_item_records.unsqueeze(0) <= 0] = float('-inf')
                probs[0, 0] = p0
                rc_probs = torch.softmax(probs, dim=1)[0]
                rc = np.random.choice(rc_probs.shape[0], p=rc_probs.cpu().detach().numpy())
                try_times = 10
                while rc != 0 and try_times > 0:
                    if last_item_records[rc] > 0:
                        right_nid = env.layouts[-1].bin_tree.node_list[nid].right
                        rr_probs = self.cls_r_rotate(target_decodes[:, -1]) / 0.3
                        rr_probs = torch.softmax(rr_probs[:, 1:], dim=1)[0]
                        rr = np.random.choice(rr_probs.shape[0], p=rr_probs.cpu().detach().numpy()) + 1

                        if env.layouts[-1].bin_tree.put_item(right_nid, rc, env.items[rc, :2], rr):
                            last_item_records[rc] -= 1
                            break

                        if rr == 1:
                            rr = 2
                        else:
                            rr = 1

                        if env.layouts[-1].bin_tree.put_item(right_nid, rc, env.items[rc, :2], rr):
                            last_item_records[rc] -= 1
                            break
                        probs[0, rc] = float('-inf')
                    rc_probs = torch.softmax(probs, dim=1)[0]
                    rc = np.random.choice(rc_probs.shape[0], p=rc_probs.cpu().detach().numpy())
                    try_times -= 1
            step += 1

        target_seq, target_len, it_records, _, _, _ = self.get_data(env)
        target_seq = target_seq.to(item_size_embedding.device)
        target_decodes = self.forward_decode(target_seq, item_size_embedding, context_embedding, end_embedding,
                                             bin_size_embedding,
                                             item_mask, it_records)
        cost_chain = nn.functional.relu(self.pred_cost(target_decodes[:, -1]))
        cost_chain[cost_chain < 0] = np.log(0.5)
        '''
        if torch.sum(last_item_records) > 1:
            cost_chain = 0
        else:
            cost_chain = 1

        print(target_seq[0, :target_len].long())
        p = target_seq[0, :target_len].long()
        with open('target_seq2.txt', 'w', encoding='utf-8') as f:
            s = ''
            for i in range(p.shape[0]):
                s += str(i) + ':  '
                for j in range(p.shape[1]):
                    s += str(p[i, j].item()) + ', '
                s += '\n'
            f.write(s)
        exit(1)
        '''
        return env, cost_chain
