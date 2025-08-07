import torch
from torch.nn import Dropout, Parameter, Softmax
from torch.nn.init import constant_, xavier_uniform_, calculate_gain
from torch_geometric.nn import Linear, SAGEConv
import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, mul, remove_diag
from torch_sparse import sum as sparsesum

class NCSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()
        self.conv1 = SAGEConv(num_features, params.hidden, bias=False)
        self.conv2 = SAGEConv(params.hidden, params.hidden, bias=False)
        self.conv3 = SAGEConv(num_features, params.hidden, bias=False)
        self.conv4 = SAGEConv(params.hidden, params.hidden, bias=False)

        self.lam = Parameter(torch.zeros(3))
        self.lam1 = Parameter(torch.zeros(2))
        self.lam2 = Parameter(torch.zeros(2))
        self.dropout = Dropout(p=params.dp1)
        self.dropout2 = Dropout(p=params.dp2)
        self.finaldp = Dropout(p=0.5)
        self.act = F.relu

        self.WX = Parameter(torch.empty(num_features, params.hidden))
        # self.lin2 = Linear(3 * params.hidden, num_classes,bias=False)
        self.lin1 = Linear(params.hidden, num_classes)
        self.args = params
        self._cached_adj_l = None
        self._cached_adj_h = None
        self._cached_adj_l_l = None
        self._cached_adj_h_h = None
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.lin1.reset_parameters()
        constant_(self.lam, 0)
        constant_(self.lam1, 0)
        constant_(self.lam2, 0)
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))

    def agg_norm(self, adj_t, mask, mtype='target'):
        if mtype == 'target':
            A_tilde = mul(adj_t, mask.view(-1, 1))
        elif mtype == 'source':
            A_tilde = mul(adj_t, mask.view(1, -1))
        if self.args.addself:
            A_tilde = fill_diag(A_tilde, 1.)
        else:
            A_tilde = remove_diag(A_tilde)
        D_tilde = sparsesum(A_tilde, dim=1)
        D_tilde_sq = D_tilde.pow_(-0.5)
        D_tilde_sq.masked_fill_(D_tilde_sq == float('inf'), 0.)
        A_hat = mul(A_tilde, D_tilde_sq.view(-1, 1))
        A_hat = mul(A_hat, D_tilde_sq.view(1, -1))
        return A_hat

    def forward(self, data):
        # x = SparseTensor.from_dense(data.x)
        x = data.x.to_dense()
        # cc_mask = torch.where(data.cc_mask <= self.threshold, 1., 0.)
        cc_mask = data.cc_mask
        # cc_mask_t = torch.unsqueeze(data.cc_mask, dim=-1)
        rev_cc_mask = torch.ones_like(cc_mask) - cc_mask
        edge_index = data.edge_index
        adj_t = SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(data.num_nodes, data.num_nodes))  # TODO

        # low_cc mask
        if data.update_cc:
            A_hat_l = self.agg_norm(adj_t, cc_mask, 'target')
            self._cached_adj_l = A_hat_l
            A_hat_l_l = self.agg_norm(adj_t, cc_mask, 'source')
            self._cached_adj_l_l = A_hat_l_l
        else:
            A_hat_l = self._cached_adj_l
            A_hat_l_l = self._cached_adj_l_l

            # high_cc mask
        if data.update_cc:
            A_hat_h = self.agg_norm(adj_t, rev_cc_mask, 'target')
            self._cached_adj_h = A_hat_h
            A_hat_h_h = self.agg_norm(adj_t, rev_cc_mask, 'source')
            self._cached_adj_h_h = A_hat_h_h
        else:
            A_hat_h = self._cached_adj_h
            A_hat_h_h = self._cached_adj_h_h

        # x和A_hat_l目前均为spares_tensor，但是在使用self.conv1的时候spares_tensor有问题
        xl = self.conv1(x, A_hat_l)
        xl = self.act(xl)
        xl = self.dropout(xl)
        xl = self.conv2(xl, A_hat_l_l)
        # high_cc partion
        xh = self.conv3(x, A_hat_h)
        xh = self.act(xh)
        xh = self.dropout(xh)
        xh = self.conv4(xh, A_hat_h_h)

        x = torch.mm(x, self.WX)

        if self.args.finalagg == 'add':
            lamxl, laml = Softmax()(self.lam1)
            lamxh, lamh = Softmax()(self.lam2)
            lamx = lamxl * cc_mask + lamxh * rev_cc_mask
            xf = lamx.view(-1, 1) * x + laml * xl + lamh * xh
            # self.embedding = xf.detach()
            xf = self.act(xf)
            xf = self.finaldp(xf)
            xf = self.lin1(xf)

        return xf