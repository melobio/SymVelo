import random
from typing import Optional
import scvelo as scv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import Dataset


class PairsDataset(Dataset):
    def __init__(self, edge_index, v, u_z, s_z, u, s, device, method, dim, c=None):
        self.edge_index = edge_index
        self.v = v
        self.s_z = s_z
        self.u_z = u_z
        self.device = device
        self.u = u
        self.s = s
        self.c = c

        def get_pairs():
            if method == 'random':
                cell_number = v.size(0)
                new_pair = torch.zeros(cell_number, 2).to(device)
                idx_old = 0
                for c in range(cell_number):
                    idx_new = (edge_index.T[:, 0] == c).nonzero()
                    idx_new = torch.max(idx_new)
                    temp_edge_index = edge_index.T[idx_old: idx_new, :]
                    new_pair[c, 0] = c
                    if random.random() < 0:
                        max_similarity = 0
                        max_end_i = 0
                        for i in temp_edge_index:
                            start_i = int(i[0].item())
                            end_i = int(i[1].item())
                            if torch.cosine_similarity(v[start_i].unsqueeze(0),
                                                       (s_z[end_i] - s_z[start_i]).unsqueeze(0)) > max_similarity:
                                max_similarity = torch.cosine_similarity(v[start_i].unsqueeze(0),
                                                                         (s_z[end_i] - s_z[start_i]).unsqueeze(0))
                                max_end_i = end_i
                        new_pair[c, 1] = max_end_i
                    else:
                        random_idx = random.randint(idx_old, idx_new - 1)
                        new_pair[c, 1] = edge_index.T[random_idx, 0]
                    idx_old = idx_new + 1

            elif method == 'randomv2':
                cell_number = v.size(0)
                sample_num = 5  # 采样数
                new_pair = torch.zeros(1, 2).to(device)
                temp_pair = torch.zeros(1, 2).to(device)
                idx_old = 0
                for c in range(cell_number):
                    idx_new = (edge_index.T[:, 0] == c).nonzero()
                    idx_new = torch.max(idx_new)
                    temp_edge_index = edge_index.T[idx_old: idx_new, :]
                    temp_pair[0, 0] = c
                    if random.random() < 0.9:
                        list_similarity = []
                        list_end_i = []
                        for i in temp_edge_index:
                            start_i = int(i[0].item())
                            end_i = int(i[1].item())
                            list_similarity.append(torch.cosine_similarity(v[start_i].unsqueeze(0),
                                                                           (s_z[end_i] - s_z[start_i]).unsqueeze(
                                                                               0)).item())
                            list_end_i.append(end_i)
                        sort_id = sorted(range(len(list_similarity)), key=lambda k: list_similarity[k], reverse=True)
                        for i in range(sample_num):
                            temp_pair[0, 1] = list_end_i[sort_id[i]]
                            new_pair = torch.cat([new_pair, temp_pair], 0)

                    else:
                        for i in range(sample_num):
                            random_idx = random.randint(idx_old, idx_new - 1)
                            temp_pair[0, 1] = edge_index.T[random_idx, 0]
                            new_pair = torch.cat([new_pair, temp_pair], 0)

                    idx_old = idx_new + 1

            elif method == 'all':
                new_pair = torch.zeros(1, 2).to(device)
                for i in edge_index.T:
                    start_i = int(i[0].item())
                    end_i = int(i[1].item())
                    if torch.cosine_similarity(v[start_i].unsqueeze(0),
                                               (s_z[end_i] - s_z[start_i]).unsqueeze(0)) > 0:
                        new_pair = torch.cat([new_pair, i.unsqueeze(0)], 0)
            if self.c is not None:
                omcis = len(self.c)
            pair_num = new_pair.size(0)  # 29704 * 2 / 2503 * 2
            if dim == 'high':
                z_dim = u.size(1)  # 2503 * 2000
                if self.c is not None:
                    new_data = torch.zeros(pair_num, z_dim, 2, 2 + omcis).to(device, dtype=torch.float64)
                else:
                    new_data = torch.zeros(pair_num, z_dim, 2, 2).to(device, dtype=torch.float64)

                for idx, i in enumerate(new_pair):
                    start_i = int(i[0].item())
                    end_i = int(i[1].item())
                    new_data[idx, :, 0, 0] = u[start_i, :]
                    new_data[idx, :, 0, 1] = s[start_i, :]
                    new_data[idx, :, 1, 0] = u[end_i, :]
                    new_data[idx, :, 1, 1] = s[end_i, :]
                    if self.c is not None:
                        for o in range(omcis):
                            new_data[idx, :, 0, 2 + o] = self.c[o][start_i, :]
                            new_data[idx, :, 1, 2 + o] = self.c[o][end_i, :]
            else:
                z_dim = u_z.size(1)  # 2503 * 100
                if self.c is not None:
                    new_data = torch.zeros(pair_num, z_dim, 2, 2 + omcis).to(device, dtype=torch.float64)
                else:
                    new_data = torch.zeros(pair_num, z_dim, 2, 2).to(device, dtype=torch.float64)
                for idx, i in enumerate(new_pair):
                    start_i = int(i[0].item())
                    end_i = int(i[1].item())
                    new_data[idx, :, 0, 0] = u_z[start_i, :]
                    new_data[idx, :, 0, 1] = s_z[start_i, :]
                    new_data[idx, :, 1, 0] = u_z[end_i, :]
                    new_data[idx, :, 1, 1] = s_z[end_i, :]
                    if self.c is not None:
                        for o in range(omcis):
                            new_data[idx, :, 0, 2 + o] = self.c[o][start_i, :]
                            new_data[idx, :, 1, 2 + o] = self.c[o][end_i, :]
            return new_pair, new_data

        self.new_pair, self.new_data = get_pairs()

    def __len__(self):
        return self.new_pair.size(0)

    def __getitem__(self, idx):
        return {
            'input': self.new_data[idx, :, 0, :].unsqueeze(1),
            'target': self.new_data[idx, :, 1, :].unsqueeze(1),
            'pair': self.new_pair[idx, :]
        }


def get_state_change_vector(edge_index, v_sym, v_velo, s_z, tensor_s, sig=1., device=None):
    # following is to calc a matrix V*V, may too large
    # cell_number = v.size(0)
    # idx_old = 0
    # p_sym = torch.zeros(cell_number, cell_number).to(device)
    #
    # for c in range(cell_number):
    #     idx_new = (edge_index.T[:, 1] == c).nonzero()
    #     if idx_new.size(0) == 0:
    #         continue
    #     idx_new = torch.max(idx_new)
    #
    #     temp_sum = []
    #     temp_edge_index = edge_index.T[idx_old: idx_new, :]
    #     for i in temp_edge_index:
    #         start_i = int(i[1].item())
    #         end_i = int(i[0].item())
    #         # print(start_i, end_i)
    #         temp_sum.append(
    #             torch.exp(sig * torch.cosine_similarity(v[start_i].unsqueeze(0),
    #                                                     (s_z[start_i] - s_z[end_i]).unsqueeze(0))).item()
    #         )
    #         p_sym[start_i, end_i] = torch.exp(
    #             sig * torch.cosine_similarity(v[start_i].unsqueeze(0), (s_z[start_i] - s_z[end_i]).unsqueeze(0)))
    #     p_sym[start_i, :] /= sum(temp_sum)
    #     idx_old = idx_new

    # follow is calc a matrix E * 1
    kl_div = torch.nn.KLDivLoss()
    start_i = edge_index[0, :].cpu().numpy() // 1
    end_i = edge_index[1, :].cpu().numpy() // 1
    cell_number = v_sym.size(0)
    p_sym = torch.zeros(edge_index.size(1)).to(device)
    p_velo = torch.zeros(edge_index.size(1)).to(device)
    loss_o = 0.
    loss_s = 0.
    for c in range(cell_number):
        pos = np.where(start_i == c)[0]
        # print(c)
        for i in pos:
            p_sym[i] = torch.cosine_similarity(v_sym[start_i[i]].unsqueeze(0), (tensor_s[end_i[i]] - tensor_s[start_i[i]]).unsqueeze(0)) * 5
            p_velo[i] = torch.cosine_similarity(v_velo[start_i[i]].unsqueeze(0), (s_z[end_i[i]] - s_z[start_i[i]]).unsqueeze(0)) * 5

        p_sym[pos[0]: pos[-1] + 1] = F.softmax(p_sym[pos[0]: pos[-1] + 1], dim=0)
        p_velo[pos[0]: pos[-1] + 1] = F.softmax(p_velo[pos[0]: pos[-1] + 1], dim=0)
        loss_o += kl_div(torch.log(p_sym.detach()[pos[0]: pos[-1] + 1]), Variable(p_velo[pos[0]: pos[-1] + 1]))
        loss_s += kl_div(torch.log(p_velo.detach()[pos[0]: pos[-1] + 1]), Variable(p_sym[pos[0]: pos[-1] + 1]))
    # loss_o = kl_div(torch.log(p_sym.detach()), Variable(p_velo))
    # loss_s = kl_div(torch.log(p_velo.detach()), Variable(p_sym))
    loss_s, loss_o = loss_s / cell_number * 10, loss_o / cell_number * 10
    return p_sym, p_velo, loss_o, loss_s


def print_eqs(expr_dict, print_prec=0.001):
    for eq in expr_dict:
        print("Eq:", eq)
        for T in range(len(expr_dict[eq])):
            print('Gene:', T + 1)
            for t in expr_dict[eq][T]:
                if abs(expr_dict[eq][T][t]) > print_prec:
                    print("%s\t%.4f" % (t, expr_dict[eq][T][t]))


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost  # , pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess data.
    This function removes poorly detected genes and minmax scales the data.
    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit
    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata