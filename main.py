import argparse
import copy
import math
from pathlib import Path
import matplotlib
import numpy
import torch
from torch.optim.lr_scheduler import StepLR
from SymNet import polypde
from SymNet.model_helper import *
import scvelo as scv
import scanpy
import scipy
import re
from torch.utils.data import DataLoader
from VeloAe.util import estimate_ld_velocity
from VeloAe.veloproj import *
from VeloAe.model import *
from utils import PairsDataset, print_eqs, get_state_change_vector, SinkhornDistance, preprocess_data
import matplotlib.pyplot as plt

matplotlib.use('agg')
parser = argparse.ArgumentParser(description='SymVelo')
# config
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default=':0')
parser.add_argument('--log-interval', type=int, default=100,
                    help='How frequntly logging training status (default: 100)')
parser.add_argument('--veloae_coef', type=float, default=1, help='coef to balance losses')
parser.add_argument('--kl', type=str, choices=['KL', 'Wass'], default='KL')
parser.add_argument('--pre_model', type=str, default='VeloAE',
                    help='# when args.frozen == True, we can choose v from scVelo or VeloAE or other(path for V)')
# SymNet
parser.add_argument('--epochs_s', type=int, default=1000, help='Epochs of SymNet training')
parser.add_argument('--lr_s', type=float, default=1e-4, help='Learning rate of SymNet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/scNTseq/',help='SymNet checkpoint save path')
parser.add_argument('--hidden_layer_s', type=int, default=2, help='Hidden layers of SymNet')
parser.add_argument('--dt', type=float, default=5e-3, help='dt')
parser.add_argument('--use_bias', type=str, choices=['True', 'False'], default='False',
                    help='True: Each linear layers have a bias as usual;'
                         'False: Use a MLP instead of all bias of linear layers of SymNet')
parser.add_argument('--optimizer', type=str, choices=['Adam', 'SDG', 'RMSprop'],
                    default='Adam')  # unfinished only default is work
parser.add_argument('--scheduler', type=str, choices=['Step', 'Lambda', 'Cosine'],
                    default='Step')  # unfinished only default is work
parser.add_argument('--batch_size', type=int, default=1500, help='SymNet batch size')
# data config
parser.add_argument('--dataset', type=str,
                    choices=['pancreas', 'dentategyrus', 'Erythroid_human', 'Erythroid_mouse', 'scEUseq', 'scNTseq',
                             'Multi', 'SHARE', 'Liver', 'GreenLeaf', 'Dyngen_Linear', 'Dyngen_Bifurcation', 'Dyngen_Trifurcation', 'Share_new'], default='Multi',
                    help='Dataset, if use other dataset, add name of dataset in choices')
parser.add_argument('--gene_number', type=int, default=2000,
                    help='The number of selected genes via data preprocessing method')
parser.add_argument('--psm', type=str, choices=['random', 'all', 'randomv2'], default='random',
                    help='Pairs sampling method. '
                         'random: 90% probability to select the neighbor cell whose direction is closest to the '
                         'velocity estimated by VeloAE and 10% probability to select a random neighbor cell as a pair;'
                         'all: Select all the neighbor cells as pairs'
                         'randomv2: Similar to random method, but select closest 5 cell as pairs')
parser.add_argument('--psd', type=str, choices=['high', 'low'], default='high',
                    help='Pairs sampling dim.(almost high, low is not dual-path)')
# VeloAE
parser.add_argument('--frozen', type=str, choices=['True', 'False'], default='False',
                    help='False: DML; True: frozen VeloAE to train SymNet')
parser.add_argument('--neighbors', type=int, default=30, help='neighbor number of knn and graph')
parser.add_argument('--pretrain_model', type=str, default='None',
                    help='VeloAE pre-trained model path')
parser.add_argument('--pretrain_odenet', type=str, default='None',
                    help='Symnet pre-trained model path, only use when the training interrupted unexpectedly')
parser.add_argument('--vis-key', type=str, default="X_umap")
parser.add_argument('--z-dim', type=int, default=100, help='Dimensionality of the low-dimensional space')
parser.add_argument('--g-rep-dim', type=int, default=100, help='Dimensionality of gene representations.')
parser.add_argument('--h-dim', type=int, default=256, help='Dimensionality of intermediate layers in MLP')
parser.add_argument('--k-dim', type=int, default=100, help='Dimensionality of keys for attention computation.')
parser.add_argument('--conv-thred', type=float, default=1e-6)
parser.add_argument('--lr_v', type=float, default=1e-6, help='Learning rate of VeloAE')
parser.add_argument('--gumbsoft_tau', type=float, default=1.0, help='Temperature param of gumbel softmax')
parser.add_argument('--aux_weight', type=float, default=1.0)
parser.add_argument('--nb_g_src', type=str, default="SU")
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--steady', type=str, choices=['True', 'False'], default='True')  # unfinished
parser.add_argument('--use_x', type=bool, default=False,
                    help="""whether or not to enroll transcriptom reads for training (default: False)."""
                    )
parser.add_argument('--refit', type=int, default=1,
                    help="""whether or not refitting veloAE, if False, need to provide a fitted model for velocity projection. (default=1)
                         """
                    )
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay strength (default 0.0)')
args = parser.parse_args()
device = torch.device("cuda" + args.device if torch.cuda.is_available() else "cpu")
seed = args.seed
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
Path(args.checkpoint).mkdir(parents=True, exist_ok=True)
channel_names = 'u, s'
hidden_layers = args.hidden_layer_s
dt = args.dt
if args.use_bias == 'True':
    use_bias = True
else:
    use_bias = False
learning_rate_s = args.lr_s
learning_rate_v = args.lr_v

epochs_s = args.epochs_s
if args.dataset == 'pancreas':
    adata = scanpy.read_h5ad('./dataset/endocrinogenesis_day15.5.h5ad')
    cluster_edges = [
        ("Ngn3 low EP", "Ngn3 high EP"),
        ("Ngn3 high EP", "Fev+"),
        ("Fev+", "Alpha"),
        ("Fev+", "Beta"),
        ("Fev+", "Delta"),
        ("Fev+", "Epsilon")]
    scv.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    k_cluster = "clusters"

elif args.dataset == 'dentategyrus':
    adata = scv.read('./data/DentateGyrus/DentateGyrus.h5ad')
    cluster_edges = [("OPC", "OL"), ("nIPC", "Neuroblast"), ("Neuroblast", "Granule immature"), ("Granule immature", "Granule mature"), ("Radial Glia-like", "Astrocytes")]
    k_cluster = "clusters"
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)


elif args.dataset == 'Erythroid_mouse':
    adata = scanpy.read_h5ad('./dataset/Erythroid_mouse.h5ad')
    sel = np.zeros(adata.n_obs, dtype=np.bool)
    sel = sel | (adata.obs.celltype == "Erythroid1").values | (adata.obs.celltype == "Erythroid2").values | (
            adata.obs.celltype == "Erythroid3").values
    sel = sel | (adata.obs.celltype == "Blood progenitors 1").values | sel | (
            adata.obs.celltype == "Blood progenitors 2").values
    adata = adata[sel]
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    cluster_edges = [("Erythroid1", "Erythroid2"), ('Erythroid2', "Erythroid3")]
    k_cluster = "celltype"

elif args.dataset == 'Erythroid_human':
    adata = scanpy.read_h5ad('./dataset/Erythroid_human.h5ad')
    sel = np.zeros(adata.n_obs, dtype=np.bool)
    sel = sel | (adata.obs.type2 == "Early Erythroid").values | (adata.obs.type2 == "Mid  Erythroid").values | (
            adata.obs.type2 == "Late Erythroid").values
    sel = sel | (adata.obs.type2 == "MEMP").values
    adata = adata[sel]
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    cluster_edges = [("Early Erythroid", "Mid  Erythroid"), ('Mid  Erythroid', "Late Erythroid")]
    k_cluster = "type2"

elif args.dataset == 'scEUseq':
    adata = scanpy.read_h5ad('./dataset/scEUseq.h5ad')
    cluster_edges = [("3", "1"), ("3", "2")]
    k_cluster = "monocle_branch_id"
    scv.utils.show_proportions(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

elif args.dataset == 'scNTseq':
    adata = scanpy.read_h5ad('./data/scNT/scNTseq.h5ad')
    cluster_edges = [("0", "15"), ("15", "30"), ("30", "60"), ("60", "120")]
    k_cluster = "time"
    scv.utils.show_proportions(adata)
    adata.obs['time'] = adata.obs.time.astype('category')
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

elif args.dataset == 'SHARE':
    adata = scanpy.read_h5ad('./dataset/SHARE-seq_TAC.h5ad')
    scv.utils.show_proportions(adata)
    # adata.obs['time'] = adata.obs.time.astype('category')
    cluster_edges = [('TAC-1', 'IRS'), ('TAC-1', 'Medulla'), ('TAC-1', 'Hair Shaft-Cuticle/Cortex')]
    k_cluster = "celltype"
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

elif args.dataset == 'Liver':
    cluster_edges = [("HSC_MPP", "MEMP"), ("MEMP", "Early Erythroid"), ("Early Erythroid", "Mid Erythroid"),
                     ("Mid Erythroid", "Late Erythroid")]
    k_cluster = "cell.labels"
    adata = scanpy.read_h5ad('./dataset/liver_small.h5ad')
    adata = adata[adata.obs['sample'] == 'FCAImmP7277561']
    scv.utils.show_proportions(adata)
    # adata.obs['time'] = adata.obs.time.astype('category')
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

elif args.dataset == 'GreenLeaf':
    adata = scanpy.read_h5ad('./dataset/greenleaf_multivelo.h5ad')
    k_cluster = "cluster"
    cluster_edges = [("Cyc.", "RG/Astro"), ("Cyc.", "mGPC/OPC"), ("Cyc.", "nIPC/ExN"), ("nIPC/ExN", "ExM"), ("ExM", "ExUp")]

elif args.dataset == 'Share_new':
    adata = scanpy.read_h5ad('./dataset/share_multivelo.h5ad')
    cluster_edges = [('TAC-1', 'IRS'), ('TAC-1', 'Medulla'), ('TAC-1', 'Hair Shaft-cuticle.cortex')]
    k_cluster = "celltype"

elif args.dataset == "Dyngen_Linear":
    adata = scanpy.read_h5ad('./dataset/synthetic_linear_mile.h5ad')
    k_cluster = 'milestone'
    cluster_edges = [('A', 'B'), ('B', 'C')]
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata = preprocess_data(adata)
    args.gene_number = adata.layers['spliced'].shape[1]
    args.g_rep_dim = adata.layers['spliced'].shape[1]

elif args.dataset == "Dyngen_Bifurcation":
    adata = scanpy.read_h5ad('./dataset/synthetic_bifurcation_mile.h5ad')
    k_cluster = 'milestone'
    cluster_edges = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'E')]
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # adata = preprocess_data(adata)
    args.gene_number = adata.layers['spliced'].shape[1]
    args.g_rep_dim = adata.layers['spliced'].shape[1]

elif args.dataset == "Dyngen_Trifurcation":
    adata = scanpy.read_h5ad('./dataset/synthetic_trifurcation_mile.h5ad')
    k_cluster = "milestone"
    cluster_edges = [('A', 'B'),
                     ('B', 'C'), ('C', 'F'),
                     ('B', 'D'), ('D', 'H'), ('E', 'H')]
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata = preprocess_data(adata)
    args.gene_number = adata.layers['spliced'].shape[1]
    args.g_rep_dim = adata.layers['spliced'].shape[1]

elif args.dataset == 'Multi':

    adata = scanpy.read_h5ad('./data/greenleaf_multiome/greenleaf_multivelo_0525.h5ad')
    cluster_edges = [("Cyc.", "RG/Astro"), ("Cyc.", "mGPC/OPC"), ("Cyc.", "nIPC/ExN"), ("nIPC/ExN", "ExM"), ("ExM", "ExUp")]
    k_cluster = "cluster"
    chromatin = adata.layers['Mc']
    chromatin = chromatin.todense()
    tensor_c = torch.DoubleTensor(chromatin).to(device)
    tensor_all = [tensor_c]
    channel_names = 'u, s, c'

print(args.g_rep_dim)
exp_metrics = {}
spliced = adata.layers['Ms']
unspliced = adata.layers['Mu']
tensor_s = torch.DoubleTensor(spliced).to(device)
tensor_u = torch.DoubleTensor(unspliced).to(device)
if type(adata.X) == type(adata.layers['Ms']):
    tensor_x = torch.DoubleTensor(adata.X).to(device)
else:
    tensor_x = torch.DoubleTensor(adata.X.toarray()).to(device)

veloae, edge_index = init_model(adata, args, device)
if args.dataset == 'scNTseq':
    veloae.criterion = nn.SmoothL1Loss()

# if args.dataset == 'Erythroid_mouse':
#     veloae.criterion = nn.SmoothL1Loss()
# veloae.criterion = nn.SmoothL1Loss()
cell_number = adata.layers['Ms'].shape[0]

if args.psd == 'low':
    odenet = polypde.POLYODE(
        dt=dt,
        channel_names=channel_names,
        hidden_layers=hidden_layers,
        bias=use_bias,
        static=True,
        gene_number=args.z_dim,
        device=device,
        cell_number=cell_number
    ).to(device)
else:
    odenet = polypde.POLYODE(
        dt=dt,
        channel_names=channel_names,
        hidden_layers=hidden_layers,
        bias=use_bias,
        static=True,
        gene_number=args.gene_number,
        device=device,
        cell_number=cell_number
    ).to(device)
dim_trans = nn.Sequential(
    # nn.Linear(2000, 1024, dtype=torch.float64),
    # nn.ReLU(),
    # nn.Linear(1024, 256, dtype=torch.float64),
    # nn.ReLU(),
    # nn.Linear(256, 100, dtype=torch.float64),
    # nn.ReLU(),
    nn.Linear(2000, 100, dtype=torch.float64),
    nn.ReLU()
).to(device)
for p in dim_trans.parameters():
    nn.init.normal_(p, 0, 1e-1)

for p in odenet.parameters():  # add random init1
    nn.init.normal_(p, 0, 1e-1)

if args.dataset == 'pancreas' or args.dataset == 'Liver' or args.dataset == 'Multi':
    for p in odenet.parameters():  # add random init
        nn.init.normal_(p, 0, 5e-3)
if args.dataset == 'Erythroid_mouse':
    for p in odenet.parameters():  # add random init
        nn.init.normal_(p, 0, 5e-2)
if args.pretrain_model != 'None':
    pretrain_model_temp = torch.load(args.pretrain_model, map_location='cpu')
# temp_s = ''
# for k in list(pretrain_model_temp.keys()):
#     if 'nns' in k:
#         if re.findall(r"\d+", k)[0] < '4' and 'bias' not in k:
#             temp_s = '.lin'
#             pretrain_model_temp[k] = pretrain_model_temp[k].T
#         else:
#             temp_s = ''
#         pretrain_model_temp[k.replace('nns.' + re.findall(r"\d+", k)[0],
#                                       'module_' + re.findall(r"\d+", k)[0] + temp_s)] = pretrain_model_temp.pop(k)
    veloae.load_state_dict(pretrain_model_temp)
veloae = veloae.to(device, dtype=torch.float64)
if args.pretrain_odenet != 'None':
    odenet.load_state_dict(torch.load(args.pretrain_odenet, map_location='cpu')).to(device)
if args.kl == 'KL':
    Dist = nn.KLDivLoss(reduction='batchmean')
else:
    Dist = SinkhornDistance(eps=.1, max_iter=100, reduction='sum')
epoch_ode_loss = []  # odenet
epoch_step_loss = []  # odenet
epoch_sparse_loss = []  # odenet
epoch_veloae_loss = []  # veloae
epoch_reg_loss = []  # veloae
epoch_rec_loss = []  # veloae
epoch_dis_loss = []
epoch_sum_loss = []
min_sym_loss = math.inf
if args.frozen == 'True':
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(odenet.parameters(), lr=learning_rate_s)  # 1e-3
    if args.scheduler == 'Step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
    veloae.eval()

    s = 1e-3
    sparsity_coef = 1
    odenet.train()
    with torch.no_grad():
        if args.pre_model == 'VeloAE':
            if args.psd == 'high':
                g_x, r_x = veloae.encoder(tensor_x, True)
                g_s, r_s = veloae.encoder(tensor_s, True)
                g_u, r_u = veloae.encoder(tensor_u, True)
                x = veloae.decoder(r_x, g_x, False)
                u_z = veloae.decoder(r_u, g_u, False)
                s_z = veloae.decoder(r_s, g_s, False)
                if args.dataset == 'Multi':
                    g_c, r_c = veloae.encoder(tensor_c, True)
                    c_z = veloae.decoder(r_c, g_c, False)
                    v = estimate_ld_velocity(g_u, g_c, device=device)
                    dataset = PairsDataset(edge_index, v, g_u, g_s, tensor_u, tensor_s, device, method=args.psm,
                                           dim=args.psd, c=tensor_all)
                else:
                    v = estimate_ld_velocity(g_s, g_u, device=device)

                    dataset = PairsDataset(edge_index, v, g_u, g_s, tensor_u, tensor_s, device, method=args.psm,
                                           dim=args.psd)
            else:
                g_x, r_x = veloae.encoder(tensor_x, True)
                g_s, r_s = veloae.encoder(tensor_s, True)
                g_u, r_u = veloae.encoder(tensor_u, True)

                x = veloae.decoder(r_x, g_x, False)
                u_z = veloae.decoder(r_u, g_u, False)
                s_z = veloae.decoder(r_s, g_s, False)
                if args.dataset == 'Multi':
                    g_c, r_c = veloae.encoder(tensor_c, True)
                    c_z = veloae.decoder(r_c, g_c, False)
                    v = estimate_ld_velocity(g_u, g_c, device=device)
                    dataset = PairsDataset(edge_index, v, g_u, g_s, g_u, g_s, device, method=args.psm, dim=args.psd,
                                           c=tensor_all)
                else:
                    v = estimate_ld_velocity(g_s, g_u, device=device)
                    dataset = PairsDataset(edge_index, v, g_u, g_s, g_u, g_s, device, method=args.psm, dim=args.psd)
        elif args.pre_model == 'scVelo':  # scVelo
            scv.tl.velocity(adata, vkey='stc_velocity', mode="stochastic")
            v = adata.layers['stc_velocity']
            v = torch.DoubleTensor(v).to(device)
            if args.dataset == 'Multi':
                dataset = PairsDataset(edge_index, v, tensor_u, tensor_s, tensor_u, tensor_s, device,
                                       method=args.psm,
                                       dim=args.psd, c=tensor_all)
            else:
                dataset = PairsDataset(edge_index, v, tensor_u, tensor_s, tensor_u, tensor_s, device,
                                       method=args.psm,
                                       dim=args.psd)
        else:
            v = numpy.load(args.pre_model)
            v = torch.DoubleTensor(v).to(device)
            if args.dataset == 'Multi':
                dataset = PairsDataset(edge_index, v, tensor_u, tensor_s, tensor_u, tensor_s, device,
                                       method=args.psm,
                                       dim=args.psd, c=tensor_all)
            else:
                dataset = PairsDataset(edge_index, v, tensor_u, tensor_s, tensor_u, tensor_s, device,
                                       method=args.psm,
                                       dim=args.psd)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)  # test

    for epoch in range(epochs_s):

        temp_epoch_sum_loss = 0.
        temp_epoch_step_loss = 0.
        temp_epoch_sparse_loss = 0.
        temp_epoch_kl_loss = 0.

        for idx, i in enumerate(dataloader):
            input, target = i['input'].to(device), i['target'].to(device)
            input, target = Variable(input), Variable(target)
            s_embedding = input[:, :, 0, 1]
            output = odenet.step(input, s_embedding)
            v_sym = ((output - input) / dt)[:, :, 0, 1]
            steploss = torch.sum(torch.mean(((output - target) / dt) ** 2, 0))

            for pp in odenet.parameters():
                p = pp.abs()
                sparse_loss = ((p < s).double() * 0.5 / s * p ** 2 + (p >= s).double() * (p - s / 2)).sum()

            # _, _, kl_ode, _ = get_state_change_vector(edge_index, v_sym, v, g_s, tensor_s, device=device)
            ode_loss = steploss + sparsity_coef * sparse_loss
            ode_loss /= len(dataset)

            optimizer.zero_grad()
            ode_loss.backward()
            optimizer.step()

            temp_epoch_sum_loss += ode_loss
            temp_epoch_step_loss += steploss
            temp_epoch_sparse_loss += sparsity_coef * sparse_loss
            # temp_epoch_kl_loss += kl_ode

        epoch_ode_loss.append(temp_epoch_sum_loss.item())
        epoch_step_loss.append(temp_epoch_step_loss.item() / len(dataset))
        epoch_sparse_loss.append(temp_epoch_sparse_loss.item() / len(dataset))
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs_s - 1:
            print('Epoch: {} / {}'.format(epoch + 1, epochs_s))
            print('Ode Loss: %.5f, Step Loss: %.5f, Sparse Loss: %.5f, KL Distance: %.5f, Lr: %.5f' % (
                temp_epoch_sum_loss,
                temp_epoch_step_loss / len(dataset),
                temp_epoch_sparse_loss / len(dataset),
                temp_epoch_kl_loss,
                scheduler.get_last_lr()[0]))
            #if temp_epoch_sum_loss < min_sym_loss:
#                 min_sym_loss = temp_epoch_sum_loss
#                 torch.save(odenet.state_dict(), './checkpoint/%s/frozen_%s_hl%d_nb_%s.pth' % (
#                     args.dataset, args.psm, args.hidden_layer_s, str(use_bias)))
            torch.save(odenet.state_dict(), '%s/odenet_%s%s_hl%d_nb_%s_ep_%s.pth' % (
            args.checkpoint, args.psm, args.psd, args.hidden_layer_s, str(use_bias),epoch+1))
            print("Model Save (%d)" % (epoch + 1))
        # if (epoch + 1) % 100 == 0:
#     if args.psd == 'high':
#         expr_dict = get_expr(odenet, ifprint=False, genes=args.gene_number)
#     else:
#         expr_dict = get_expr(odenet, ifprint=False, genes=args.psd)

#     print_eqs(expr_dict)
#     with open("expr_dict.pkl", "wb") as tf:
#         pickle.dump(expr_dict, tf)

else:
    if args.optimizer == 'Adam':
        if args.dataset == 'Multi':
            optimizer_s = torch.optim.Adam([
                {'params': odenet.parameters()},
                {'params': dim_trans.parameters()}
            ], lr=learning_rate_s)  # 1e-3
        else:
            optimizer_s = torch.optim.Adam(odenet.parameters(), lr=learning_rate_s, weight_decay=1e-2)
        optimizer_v = torch.optim.Adam(veloae.parameters(), lr=learning_rate_v)  # 1e-6
    if args.scheduler == 'Step':
        scheduler_s = StepLR(optimizer_s, step_size=20, gamma=0.95)
        scheduler_v = StepLR(optimizer_v, step_size=20, gamma=0.95)

    s = 1e-3
    sparsity_coef = 1
    veloae_coef = args.veloae_coef
    odenet.train()
    veloae.train()
    for epoch in range(epochs_s):
        temp_epoch_sum_loss = 0.
        temp_epoch_ode_loss = 0.
        temp_epoch_step_loss = 0.
        temp_epoch_sparse_loss = 0.
        temp_epoch_veloae_loss = 0.
        temp_epoch_rec_loss = 0.
        temp_epoch_reg_loss = 0.
        temp_epoch_dis_loss = 0.

        if epoch % 1 == 0:
            with torch.no_grad():
                if args.psd == 'high':
                    g_x, r_x = veloae.encoder(tensor_x, True)
                    g_s, r_s = veloae.encoder(tensor_s, True)
                    g_u, r_u = veloae.encoder(tensor_u, True)
                    if args.dataset == 'Multi':
                        g_c, r_c = veloae.encoder(tensor_c, True)
                        v = estimate_ld_velocity(g_u, g_c, device=device)
                        dataset = PairsDataset(edge_index, v, g_u, g_s, tensor_u, tensor_s, device, method=args.psm,
                                               dim=args.psd, c=tensor_all)
                    else:
                        v = estimate_ld_velocity(g_s, g_u, device=device)
                        dataset = PairsDataset(edge_index, v, g_u, g_s, tensor_u, tensor_s, device, method=args.psm,
                                               dim=args.psd)

                else:
                    g_x, r_x = veloae.encoder(tensor_x, True)
                    g_s, r_s = veloae.encoder(tensor_s, True)
                    g_u, r_u = veloae.encoder(tensor_u, True)
                    if args.dataset == 'Multi':
                        g_c, r_c = veloae.encoder(tensor_c, True)
                        v = estimate_ld_velocity(g_u, g_c, device=device)
                        dataset = PairsDataset(edge_index, v, g_u, g_s, g_u, g_s, device, method=args.psm, dim=args.psd,
                                               c=tensor_all)
                    else:
                        v = estimate_ld_velocity(g_s, g_u, device=device)
                        dataset = PairsDataset(edge_index, v, g_u, g_s, g_u, g_s, device, method=args.psm, dim=args.psd)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)  # test
                # if args.steady == 'True':  # unfinished 稳态假设基于每个细胞的每个基因 无法限定某个细胞
                #     steady_mask = get_mask_pt(u_z, s_z, perc=[5, 95], device=device)

        rec_loss = veloae(tensor_s) + veloae(tensor_u)
        x, raw_x = veloae.encoder(tensor_x, True)
        s_z, raw_s = veloae.encoder(tensor_s, True)
        u_z, raw_u = veloae.encoder(tensor_u, True)
        if args.psd == 'high':
            if args.dataset == 'Multi':
                c_z, raw_c = veloae.encoder(tensor_c, True)
                _, gamma, reg_loss = leastsq_pt(
                    u_z, c_z,
                    fit_offset=True,
                    perc=[5, 95],
                    device=device,
                    norm=False
                )
                v_veloae = c_z - gamma * u_z

            else:
                _, gamma, reg_loss = leastsq_pt(
                    s_z, u_z,
                    fit_offset=True,
                    perc=[5, 95],
                    device=device,
                    norm=False
                )
                v_veloae = u_z - gamma * s_z

        else:
            _, gamma, reg_loss = leastsq_pt(
                s_z, u_z,
                fit_offset=True,
                perc=[5, 95],
                device=device,
                norm=False
            )
            v_veloae = u_z - gamma * s_z

        rec_loss *= veloae_coef
        reg_loss = torch.sum(reg_loss)
        reg_loss *= veloae_coef
        kl_veloae_sum = 0.
        for idx, i in enumerate(dataloader):
            input, target = i['input'].to(device), i['target'].to(device)
            input, target = Variable(input), Variable(target)
            s_embedding = input[:, :, 0, 1]
            output = odenet.step(input, s_embedding)
            v_sym = ((output - input) / dt)[:, :, 0, 1]
            steploss = torch.sum(torch.mean(((output - target) / dt) ** 2, 0)) * 20
            # steploss = 0
            for pp in odenet.parameters():
                p = pp.abs()
                sparse_loss = ((p < s).double() * 0.5 / s * p ** 2 + (p >= s).double() * (p - s / 2)).sum()
            sparse_loss *= sparsity_coef

            if args.dataset == 'Multi':
                # low_v_sym = dim_trans(v_sym)
                # distance_loss = nn.MSELoss()
                # kl_ode = distance_loss(low_v_sym.detach(), Variable(v_veloae[idx * args.batch_size: idx * args.batch_size + low_v_sym.size(0), :]))
                # kl_veloae = distance_loss(v_veloae[idx * args.batch_size: idx * args.batch_size + low_v_sym.size(0), :].detach(), Variable(low_v_sym))
                _, _, kl_ode, kl_veloae = get_state_change_vector(edge_index, v_sym, v_veloae, s_z, tensor_s,
                                                                  device=device)
            else:
                if args.psd == 'high':
                    _, _, kl_ode, kl_veloae = get_state_change_vector(edge_index, v_sym, v_veloae, s_z, tensor_s,
                                                                      device=device)
                else:
                    _, _, kl_ode, kl_veloae = get_state_change_vector(edge_index, v_sym, v_veloae, s_z, s_z,
                                                                      device=device)

            # if args.dataset == 'pancreas':
            #     if epoch < 50:
            #         ode_loss = sparse_loss + kl_ode * len(dataset)
            # else:
            #     ode_loss = steploss + sparse_loss + kl_ode * len(dataset)
            ode_loss = steploss + sparse_loss + kl_ode * len(dataset)
            ode_loss /= len(dataset)
            optimizer_s.zero_grad()
            ode_loss.backward()
            optimizer_s.step()
            kl_veloae_sum += kl_veloae

            temp_epoch_ode_loss += ode_loss
            temp_epoch_step_loss += steploss
            temp_epoch_sparse_loss += sparse_loss
            temp_epoch_dis_loss += kl_ode + kl_veloae

        veloae_loss = rec_loss + reg_loss + kl_veloae_sum
        optimizer_v.zero_grad()
        veloae_loss.backward()
        # sum_loss.backward()
        optimizer_v.step()

        epoch_ode_loss.append(temp_epoch_ode_loss.item())
        epoch_sparse_loss.append(temp_epoch_sparse_loss.item() / len(dataset))
        epoch_rec_loss.append(rec_loss.item())
        epoch_reg_loss.append(reg_loss.item())
        epoch_veloae_loss.append(rec_loss.item() + reg_loss.item())
        epoch_dis_loss.append(temp_epoch_dis_loss.item())
        epoch_sum_loss.append(temp_epoch_ode_loss.item() + veloae_loss.item())
        scheduler_s.step()
        scheduler_v.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs_s - 1:
            print('Epoch: {} / {}'.format(epoch + 1, epochs_s))
            print('VeloAE Loss: %.5f, Reg Loss: %.5f, Rec Loss: %.5f, Distance: %.5f' % (
                veloae_loss,
                reg_loss,
                rec_loss,
                temp_epoch_dis_loss))
            print('Ode Loss: %.5f, Step Loss: %.5f, Sparse Loss: %.5f, Lr(s | v): %.5f | %.5f' % (
                temp_epoch_ode_loss,
                temp_epoch_step_loss / len(dataset),
                temp_epoch_sparse_loss / len(dataset),
                scheduler_s.get_last_lr()[0],
                scheduler_v.get_last_lr()[0]))
            #if temp_epoch_dis_loss < min_sym_loss:
            min_sym_loss = temp_epoch_dis_loss
            torch.save(odenet.state_dict(), '%s/odenet_%s%s_hl%d_nb_%s_ep_%s.pth' % (
                args.checkpoint, args.psm, args.psd, args.hidden_layer_s, str(use_bias),epoch+1))
            torch.save(veloae.state_dict(),
                       '%s/veloae_%s%s_hl%d_nb_%s_ep_%s.pth' % (
                           args.checkpoint, args.psm, args.psd, args.hidden_layer_s, str(use_bias),epoch+1))
            print("Model Save (%d)" % (epoch + 1))
        # if (epoch + 1) % 100 == 0:
    # if args.psd == 'high':
    #     expr_dict = get_expr(odenet, ifprint=False, genes=args.gene_number)
    # else:
    #     expr_dict = get_expr(odenet, ifprint=False, genes=args.z_dim)
    # with open("./checkpoint/%s/expr_dict.pkl" % args.dataset, "wb") as tf:
    #     pickle.dump(expr_dict, tf)
    # print_eqs(expr_dict)

x = [i + 1 for i in range(len(epoch_ode_loss))]
# x = x[50:]

plt.plot(x, epoch_ode_loss, label='Ode Loss')
if args.frozen == 'False':
    plt.plot(x, epoch_veloae_loss, label='VeloAE loss')
    plt.plot(x, epoch_dis_loss, label='KL/Wass loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss_Epoch")
plt.legend()
plt.savefig('%s/Loss with Epoch' % args.checkpoint)
plt.close()
