# -*- coding: utf-8 -*-
"""VeloAutoencoder module.

This module contains the veloAutoencoder and its ablation configurations.

"""
import torch
import numpy as np

from torch import nn
from torch_geometric.nn import GCNConv, Sequential, GATConv
from torch.nn import functional as F


class Encoder(nn.Module):
    """Encoder
    
    """

    def __init__(self,
                 in_dim,
                 z_dim,
                 edge_index,
                 edge_weight,
                 h_dim=256
                 ):
        """
        Args:
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            h_dim (int): dimensionality of intermediate layers in MLP
            
        """
        super(Encoder, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.fn = nn.Sequential(
            nn.Linear(in_dim, h_dim, bias=True, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, bias=True, dtype=torch.float64),
            nn.GELU(),
        )
        self.gc = Sequential("x, edge_index",
                             [(GATConv(h_dim, z_dim, edge_dim=1),
                               "x, edge_index -> x"),
                              #   nn.BatchNorm1d(z_dim),
                              nn.GELU(),
                              (GATConv(z_dim, z_dim, edge_dim=1),
                               "x, edge_index -> x"),
                              #   nn.BatchNorm1d(z_dim),
                              nn.GELU(),
                              nn.Linear(z_dim, z_dim)]
                             )
        self.gat = GATConv(h_dim, z_dim)
        self.gen = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=True)
        )

    def forward(self, x, return_raw=False, edge_index=None):
        z = self.fn(x)
        if edge_index is None:
            z = self.gc(z, self.edge_index)
        else:
            z = self.gc(z, edge_index)

        if return_raw:
            return self.gen(z), z
        return self.gen(z)


class Decoder(nn.Module):
    """Decoder
    
    """

    def __init__(self,
                 z_col_dim,
                 G_rep=None,
                 n_genes=None,
                 g_rep_dim=None,
                 k_dim=32,
                 h_dim=256,
                 gb_tau=1.0,
                 device=None
                 ):
        """
        Args:
            z_col_dim (int): size of column vectors in Z.
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            n_genes (int): number of genes.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            k_dim (int): dimensionality of keys for attention computation.
            h_dim (int): dimensionality of intermediate layers of MLP.
            gb_tau (float): temperature param of gumbel softmax
            device (torch.device): torch device object.
            
        """
        super(Decoder, self).__init__()
        self.device = device
        if not G_rep is None:
            g_rep_dim = G_rep.shape[-1]
        self.key_Z = nn.Sequential(
            nn.Linear(z_col_dim, h_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(h_dim, k_dim, dtype=torch.float64)
        )
        self.key_G = nn.Sequential(
            nn.Linear(g_rep_dim, k_dim, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(k_dim, k_dim, dtype=torch.float64)
        )
        self.G_rep = self._init_G_emb(n_genes, g_rep_dim) if G_rep is None else torch.DoubleTensor(G_rep).to(device)
        self.attn = Attention(gb_tau)

    def _init_G_emb(self, n_genes, rep_dim):
        embedder = torch.empty(n_genes, rep_dim, dtype=torch.float64)
        nn.init.xavier_normal_(embedder)
        return nn.Parameter(embedder).to(self.device)

    def forward(self, raw_Z, gen_Z, return_attn=False):
        Z = raw_Z.T
        key = self.key_Z(Z)
        query = self.key_G(self.G_rep)
        X_hat_means, p_attn = self.attn(query, key, gen_Z.T, device=self.device)
        if return_attn:
            return X_hat_means.T, p_attn.T
        return X_hat_means.T


class Attention(nn.Module):
    """Compute 'Scaled Dot Product Attention'.
    
    """

    def __init__(self, gb_tau=1.0):
        super(Attention, self).__init__()
        self.gb_tau = gb_tau

    def forward(self, query, key, value, device=None):
        """
        Args:
            query (torch.FloatTensor): query vectors identifying the gene profiles to be reconstructed.
            key (torch.FloatTensor): key vectors identifying the latent profiles to be attended to.
            value (torch.FloatTensor): Z.
            device (torch.device): torch device object.
            
        Returns:
            FloatTensor: shape (n_genes, n_cells), reconstructed input
            FloatTensor: shape (n_genes, z_dim), gene by attention distribution matrix
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        p_attn = F.gumbel_softmax(scores, tau=self.gb_tau, hard=False, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class AblationEncoder(nn.Module):
    """Encoder for Ablation Study
    
    """

    def __init__(self,
                 in_dim,
                 z_dim,
                 h_dim=256,
                 batchnorm=False
                 ):
        super(AblationEncoder, self).__init__()
        if batchnorm:
            self.fn = nn.Sequential(
                nn.Linear(in_dim, h_dim, bias=True),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Linear(h_dim, z_dim, bias=True),
                nn.LayerNorm(z_dim),
                nn.GELU(),
            )
        else:
            self.fn = nn.Sequential(
                nn.Linear(in_dim, h_dim, bias=True),
                nn.GELU(),
                nn.Linear(h_dim, z_dim, bias=True),
                nn.GELU(),
            )

    def forward(self, x):
        z = self.fn(x)
        return z


class AblationDecoder(nn.Module):
    """Decoder for Ablation Study.
    
    """

    def __init__(self,
                 z_dim,
                 out_dim,
                 h_dim=256
                 ):
        super(AblationDecoder, self).__init__()
        """
        """
        self.fc = nn.Sequential(
            nn.Linear(z_dim, out_dim),
        )

    def forward(self, Z):
        return self.fc(Z)


class AblationCohAgg(nn.Module):
    """Ablation with only Cohort Aggregation.
    
    """

    def __init__(self,
                 edge_index,
                 edge_weight,
                 in_dim,
                 z_dim,
                 h_dim=256,
                 device=None
                 ):
        """
        Args:
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            h_dim (int): dimensionality of intermediate layers in MLP
            device (torch.device): torch device object.
            
        """
        super(AblationCohAgg, self).__init__()
        self.device = device
        self.encoder = Encoder(in_dim, z_dim, edge_index, edge_weight, h_dim=h_dim)
        self.decoder = AblationDecoder(z_dim, in_dim, h_dim)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, X):
        z = self.encoder(X)
        X_hat = self.decoder(z)
        return self.criterion(X_hat, X)


class AblationAttComb(nn.Module):
    """Ablation with only Attentive Combination.
    
    """

    def __init__(self,
                 z_dim,
                 n_genes,
                 n_cells,
                 h_dim=256,
                 k_dim=100,
                 G_rep=None,
                 g_rep_dim=None,
                 gb_tau=1.0,
                 batchnorm=False,
                 device=None
                 ):
        """
        
        Args:
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            n_genes (int): number of genes
            n_cells (int): number of cells
            h_dim (int): dimensionality of intermediate layers in MLP
            k_dim (int): dimensionality of keys for attention computation
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            gb_tau (float): temperature parameter for gumbel softmax,
            device (torch.device): torch device object.
            
        """
        super(AblationAttComb, self).__init__()
        self.device = device
        self.encoder = AblationEncoder(n_genes, z_dim, h_dim=h_dim, batchnorm=batchnorm)
        self.trans_z = nn.Linear(z_dim, z_dim, bias=True)
        self.decoder = Decoder(n_cells, G_rep, n_genes, g_rep_dim, k_dim, h_dim, gb_tau, device)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, X):
        z = self.encoder(X)
        gen_z = self.trans_z(z)
        X_hat = self.decoder(z, gen_z, False)
        return self.criterion(X_hat, X)


class VeloAutoencoder(nn.Module):
    """Proposed VeloAutoencoder with both mechanisms.
        
    """

    def __init__(self,
                 edge_index,
                 edge_weight,
                 in_dim,
                 z_dim,
                 n_genes,
                 n_cells,
                 h_dim=256,
                 k_dim=32,
                 G_rep=None,
                 g_rep_dim=None,
                 gb_tau=1.0,
                 device=None
                 ):
        """
        
        Args:
            edge_index (LongTensor): shape (2, ?), edge indices
            edge_weight (FloatTensor): shape (?), edge weights.
            in_dim (int): dimensionality of the input
            z_dim (int): dimensionality of the low-dimensional space
            n_genes (int): number of genes
            n_cells (int): number of cells
            h_dim (int): dimensionality of intermediate layers in MLP
            k_dim (int): dimensionality of keys for attention computation
            G_rep (np.ndarry): representation for genes, e.g. PCA over gene profiles.
            g_rep_dim (int): dimensionality of gene representations.
                # Either G_rep or (n_genes, g_rep_dim) should be provided.
                # priority is given to G_rep.
            gb_tau (float): temperature parameter for gumbel softmax,
            device (torch.device): torch device object.
            
        """
        super(VeloAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(in_dim, z_dim, edge_index, edge_weight, h_dim=h_dim)
        self.decoder = Decoder(n_cells, G_rep, n_genes, g_rep_dim, k_dim, h_dim, gb_tau, device)
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.SmoothL1Loss()

    def forward(self, X):
        gen_z, raw_z = self.encoder(X, True)
        X_hat = self.decoder(raw_z, gen_z, False)
        return self.criterion(X_hat, X)


def get_mask_pt(x, y=None, perc=[5, 95], device=None):
    """Mask for matrix elements selected for regression 
        (adapt from scVelo)

    Args:
        x (Tensor): Splicing counts projection
        y (Tensor): Unsplicing counts projection
        perc (int): percentile
    return:
        mask (Tensor): bool matrix
    """
    with torch.no_grad():
        xy_norm = torch.clone(x)
        if y is not None:
            y = torch.clone(y)
            xy_norm = xy_norm / torch.clip(torch.max(xy_norm, axis=0).values - torch.min(xy_norm, axis=0).values, 1e-3,
                                           None)
            xy_norm += y / torch.clip(torch.max(y, axis=0).values - torch.min(y, axis=0).values, 1e-3, None)
        if isinstance(perc, int):
            mask = xy_norm >= torch.quantile(xy_norm, perc / 100, dim=0)
        else:
            lb, ub = torch.quantile(xy_norm, torch.DoubleTensor(perc).to(device) / 100, dim=0, keepdim=True)
            mask = (xy_norm <= lb) | (xy_norm >= ub)
    return mask


def prod_sum_obs_pt(A, B):
    """dot product and sum over axis 0 (obs) equivalent to np.sum(A * B, 0)"""
    return torch.einsum("ij, ij -> j", A, B) if A.ndim > 1 else (A * B).sum()


def sum_obs_pt(A):
    """summation over axis 0 (obs) equivalent to np.sum(A, 0)"""
    return torch.einsum("ij -> j", A) if A.ndim > 1 else torch.sum(A)


def leastsq_pt(x, y, fit_offset=True, constraint_positive_offset=False,
               perc=None, device=None, norm=False):
    """Solves least squares X*b=Y for b. (adatpt from scVelo)
    
    Args:
        x (Tensor): low-dim splicing projection
        y (Tensor): low-dim unsplicing projection
        fit_offset (bool): whether fit offset
        constraint_positive_offset (bool): whether make non-negative offset
        perc (int or list of int): percentile threshold for points in regression
        device (torch.device): GPU/CPU device object

    returns:
        fitted offset, gamma and MSE losses
    """

    """Solves least squares X*b=Y for b."""
    if norm:
        x = (x - torch.mean(x, dim=0, keepdim=True)) / torch.std(x, dim=0, keepdim=True)
        y = (y - torch.mean(y, dim=0, keepdim=True)) / torch.std(y, dim=0, keepdim=True)
        x = torch.clamp(x, -1, 1)
        y = torch.clamp(y, -1, 1)

    if perc is not None:
        if not fit_offset:
            perc = perc[1]
        weights = get_mask_pt(x, y, perc=perc, device=device)
        x, y = x * weights, y * weights
    else:
        weights = None

    xx_ = prod_sum_obs_pt(x, x)
    xy_ = prod_sum_obs_pt(x, y)
    n_obs = x.shape[0] if weights is None else sum_obs_pt(weights)

    if fit_offset:

        x_ = sum_obs_pt(x) / n_obs
        y_ = sum_obs_pt(y) / n_obs
        gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_ ** 2)
        offset = y_ - gamma * x_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset = torch.clip(offset, 0, None)
    else:
        gamma = xy_ / xx_
        offset = torch.zeros(x.shape[1]).to(device) if x.ndim > 1 else 0
    nans_offset, nans_gamma = torch.isnan(offset), torch.isnan(gamma)
    if torch.any(nans_offset) or torch.any(nans_gamma):
        offset[torch.isnan(offset)], gamma[torch.isnan(gamma)] = 0, 0

    loss = torch.square(y - x * gamma.view(1, -1) - offset)
    if perc is not None:
        loss = loss * weights
    loss = sum_obs_pt(loss) / n_obs
    return offset, gamma, loss


def leastsq_generalized(x, y, fit_offset=True, constraint_positive_offset=False,
                        perc=None, device=None, norm=False):
    if norm:
        x = (x - torch.mean(x, dim=0, keepdim=True)) / torch.std(x, dim=0, keepdim=True)
        y = (y - torch.mean(y, dim=0, keepdim=True)) / torch.std(y, dim=0, keepdim=True)
        x = torch.clamp(x, -1, 1)
        y = torch.clamp(y, -1, 1)
    if perc is not None:
        if not fit_offset:
            perc = perc[1]
        weights = get_mask_pt(x, y, perc=perc, device=device)
        x, y = x * weights, y * weights
    else:
        weights = None
    yp = y + 2 * y * x
    xp = 2 * x ** 2 - x
    n_obs = x.shape[0] if weights is None else sum_obs_pt(weights)

    xx_ = torch.sum(x * x, 0) / n_obs
    xy_ = torch.sum(x * y, 0) / n_obs
    xpxp_ = torch.sum(xp * xp, 0) / n_obs
    xpyp_ = torch.sum(xp * yp, 0) / n_obs

    if fit_offset:
        x_ = torch.sum(x, 0) / n_obs
        y_ = torch.sum(y, 0) / n_obs
        xp_ = torch.sum(xp, 0) / n_obs
        yp_ = torch.sum(yp, 0) / n_obs
        gamma = (xy_ + xpyp_ - x_ * y_ - xp_ * yp_) / (xx_ + xpxp_ - x_ ** 2 - xp_ ** 2)
        offset1 = y_ - gamma * x_
        offset2 = yp_ - gamma * xp_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset1 < 0 or offset2 < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset1 = torch.clip(offset1, 0, None)
            offset2 = torch.clip(offset2, 0, None)
    else:
        gamma = (xy_ + xpyp_) / (xx_ + xpxp_)
        offset1 = torch.zeros(x.shape[1]).to(device) if x.ndim > 1 else 0
        offset2 = torch.zeros(x.shape[1]).to(device) if x.ndim > 1 else 0

    nans_offset1, nans_offset2, nans_gamma = torch.isnan(offset1), torch.isnan(offset2), torch.isnan(gamma)
    if torch.any(nans_offset1) or torch.any(nans_gamma) or torch.any(nans_offset2):
        offset1[torch.isnan(offset1)], gamma[torch.isnan(gamma)], offset2[torch.isnan(offset2)] = 0, 0, 0
    # offset1[torch.isnan(offset1)], offset2[torch.isnan(offset2)], gamma[torch.isnan(gamma)] = 0, 0, 0
    loss = torch.square(y - x * gamma.view(1, -1) - offset1) + torch.square(yp - xp * gamma.view(1, -1) - offset2)
    if perc is not None:
        loss = loss * weights
    loss = torch.sum(loss, 0) / n_obs
    return offset1, offset2, gamma, loss