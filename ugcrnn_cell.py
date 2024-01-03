import numpy as np
import torch
from lib import utils
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import init
from .custom_transformer import *
from .layer_params import *


class UGCGRUCell(nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, is_trans=True, is_bnn=True,
                 prior_mu=0, prior_sigma=0.05,trans_drop=0.3,att_type='pool_att',
                 fwd_type='sq'):
        """

        :param num_units:   ？？？表示隐层的个数
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__() # 是安装GCGRUCELL的父类Module的初始化方式进行初始化
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self.adj_mx = adj_mx
        self._num_units = num_units # 隐层的个数
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru # 是否使用图卷积计算重置门r 和 更新门u
        self.transformer = MeSFormer(dropout=trans_drop,forwordatt_type=fwd_type,atten_type=att_type, )
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx, lambda_max=None))
        elif filter_type == "attention":
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx, lambda_max=None)) # 没有修改
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
        elif filter_type == "dual_random_walk": # 双随机游走
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc') # 这里的self就是DCGRUCell
        self._gconv_params = LayerParams(self, 'gconv')

        self.is_trans = is_trans
        self.is_bnn=is_bnn
        # print(is_bnn,'2222222222222222222')
        if is_bnn:
            self.prior_mu = prior_mu  # 0
            self.prior_sigma = prior_sigma  # 0.005
            self.prior_log_sigma = math.log(prior_sigma)

            self._gconv_mu_params = LayerParams(self, 'bnn_gconv_mu',prior_log_sigma=self.prior_log_sigma)
            self._gconv_log_sigma_params = LayerParams(self, 'bnn_gconv_log_sigma', prior_log_sigma=self.prior_log_sigma)


            self.register_buffer('weight_eps', None)
            self.register_buffer('bias_eps', None)

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()  # 先转换成稀疏矩阵的形式(位置, 值)
        indices = np.column_stack((L.row, L.col))  #然后合并行索引和列索引位置
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]  #现根据列索引进行排序，之后根据行索引进行排序。
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)  #将数据转换为torch的稀疏矩阵表示
        return L

    @staticmethod
    def cheb_polynomial(laplacian, K):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([K, N, N], device=device)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=device)  # 0阶的切比雪夫多项式为单位阵

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2: # 1阶切比雪夫多项式就是拉普拉斯矩阵 L 本身
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2] #切比雪夫多项式的递推式:T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(adj):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        D = torch.diag(torch.sum(adj, dim=-1) ** (-1 / 2))
        D = torch.where(torch.isinf(D), torch.full_like(D, 0), D)
        L = torch.eye(adj.size(0), device=adj.device) - torch.mm(torch.mm(D, adj), D) # L = I - D^-1/2 * A * D^-1/2
        return L


    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.带有图卷积的门控循环单元(GRU)
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            if self.is_bnn:
                fn = self._wugconv(inputs, hx, output_size, bias_start=1.0, is_trans=self.is_trans)  # can use other graph conv?
            else:
                fn = self._gconv(inputs, hx, output_size, bias_start=1.0, is_trans=self.is_trans) # can use other graph conv?


        else:
            if self.is_bnn:
                fn = self._wugconv(inputs, hx, output_size, bias_start=1.0, is_trans=self.is_trans)
            else:
                fn = self._dconv(inputs, hx, output_size, bias_start=1.0)
        # print(fn.size,'8888888888888')
        # print(fn,'99999999999999999999')
        value = torch.sigmoid(fn)


        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        if self.is_bnn:
            c_gcn = self._wugconv(inputs, r * hx, self._num_units, is_trans=self.is_trans)  # is wuconv?
        else:
            c_gcn = self._gconv(inputs, r * hx, self._num_units, is_trans=self.is_trans) # can use other graph conv?

        if self._activation is not None:
            c_gcn = self._activation(c_gcn)

        if self.is_trans:
            c_gcn = self.transformer(c_gcn)#和上面的trans只取其一
        # print(c_gcn.size(),'2222222222222222222')
        new_state = u * hx + (1.0 - u) * c_gcn  # 参考GRU中的
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):  # 普通GRU
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.matmul(inputs_and_state, weights)
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    # dcrnn
    def _dconv(self, inputs, state, output_size, bias_start=0.0):   #
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    # gcn
    def _gconv(self, inputs, state, output_size, bias_start=0.0,is_trans=True):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # print(inputs_and_state.size(),'2222222222222')
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)

        for support in self._supports:
            x1 = torch.sparse.mm(support, x0) # L * X
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        # weights = sparse_w.apply(weights, 0.05)
        weights = nn.Dropout(0)(weights)
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases  # hx = XW + b
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        if is_trans:
            return torch.reshape(x, [batch_size, self._num_nodes, output_size])

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


        # gcn
    def _wugconv(self, inputs, state, output_size, bias_start=1.0,is_trans=True):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # print(inputs_and_state.size(),'2222222222222')
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, input_size * batch_size)

        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)  # L * X
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])


        weight_mu = self._gconv_mu_params.get_bnn_weights((input_size * num_matrices, output_size),is_mu=True)  # 初始化不确定参数
        weight_log_sigma = self._gconv_log_sigma_params.get_bnn_weights((input_size * num_matrices, output_size),is_log_sigma=True)
        if self.weight_eps is None:
            weights = weight_mu + torch.exp(weight_log_sigma) * torch.randn_like(weight_log_sigma)
        else:
            weights = weight_mu + torch.exp(weight_log_sigma) * self.weight_eps
        # weights = sparse_w.apply(weights, 0.05)
        # weights = nn.Dropout(0)(weights)
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases_mu = self._gconv_mu_params.get_bnn_biases(output_size, is_mu=True)
        biases_log_sigma = self._gconv_log_sigma_params.get_bnn_biases(output_size, is_log_sigma=True)
        if self.bias_eps is None:
            biases = biases_mu + torch.exp(biases_log_sigma) * torch.randn_like(biases_log_sigma)
        else:
            biases = biases_mu + torch.exp(biases_log_sigma) * self.bias_eps
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        if is_trans:
            return torch.reshape(x, [batch_size, self._num_nodes, output_size])

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    # chebnet
    def _chebconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)

        L = self.get_laplacian(torch.tensor(self.adj_mx, device=device)) # [N, N]
        mul_L = self.cheb_polynomial(L, K=2) # [K, N, N]
        # print(mul_L.shape, x0.shape) # torch.Size([2, 82, 82]) torch.Size([82, 8320])
        for _ in range(len(self._supports)):
            x1 = torch.matmul(mul_L, x0) # (k, num_nodes, input_size * batch_size)
            x1 = torch.sum(x1, dim=0) # (num_nodes, input_size * batch_size)
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])





