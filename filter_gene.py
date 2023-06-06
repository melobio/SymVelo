import argparse
from pathlib import Path
import matplotlib
import scanpy
import scvelo as scv
import sympy
import torch

from SymNet import polypde
from SymNet.model_helper import *
from VeloAe.eval_util import evaluate
from VeloAe.model import *
from VeloAe.util import new_adata

matplotlib.use('agg')
parser = argparse.ArgumentParser(description='SymVelo')
# config
parser.add_argument('--filter_number', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, choices=[':0', ':1', ':2', ':3'], default=':0')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how frequntly logging training status (default: 100)')
parser.add_argument('--veloae_coef', type=float, default=1, help='loss coef')
parser.add_argument('--kl', type=str, choices=['KL', 'Wass'], default='KL')
# SymNet
parser.add_argument('--ode_type', type=str, choices=['Linear', 'Quadratic'], default='Linear')
parser.add_argument('--epochs_s', type=int, default=1000)
parser.add_argument('--lr_s', type=float, default=1e-4)
parser.add_argument('--hidden_layer_s', type=int, default=2)
parser.add_argument('--dt', type=float, default=5e-3)
parser.add_argument('--use_bias', type=str, choices=['True', 'False'], default='True',
                    help='false mean MLP instead of bias')
parser.add_argument('--optimizer', type=str, choices=['Adam', 'SDG', 'RMSprop'], default='Adam')  # unfinished
parser.add_argument('--scheduler', type=str, choices=['Step', 'Lambda', 'Cosine'], default='Step')  # unfinished
parser.add_argument('--checkpoint', type=str)

# data config
parser.add_argument('--dataset', type=str,
                    choices=['pancreas', 'dentategyrus', 'Erythroid_human', 'Erythroid_mouse', 'scEUseq', 'scNTseq',
                             'Multi'], default='scEUseq')  # unfinished
parser.add_argument('--gene_number', type=int, default=2000)
parser.add_argument('--psm', type=str, choices=['random', 'all', 'randomv2'], default='randomv2',
                    help='pairs sampling method.')
parser.add_argument('--psd', type=str, choices=['high', 'low'], default='high', help='pairs sampling dim (100 / 2000)')
# VeloAE
parser.add_argument('--frozen', type=str, choices=['True', 'False'], default='True', help='False means DML')
parser.add_argument('--encoder', type=str, choices=['GCN', 'GAT', 'SGCN'], default='GCN')
parser.add_argument('--neighbors', type=int, default=30)
parser.add_argument('--pretrain_model', type=str, default=Path('./pretrain_model/test.cpt'))
parser.add_argument('--pretrain_odenet', type=str, default=Path('./checkpoint/scEUseq/odenet_randomhigh_hl2_nb_True.pth'))
parser.add_argument('--vis-key', type=str, default="X_umap")
parser.add_argument('--z-dim', type=int, default=100)
parser.add_argument('--g-rep-dim', type=int, default=100)
parser.add_argument('--h-dim', type=int, default=256)
parser.add_argument('--k-dim', type=int, default=100)
parser.add_argument('--conv-thred', type=float, default=1e-6)
parser.add_argument('--epochs_v', type=int, default=20000)
parser.add_argument('--lr_v', type=float, default=1e-6)
parser.add_argument('--gumbsoft_tau', type=float, default=1.0)
parser.add_argument('--aux_weight', type=float, default=1.0)
parser.add_argument('--nb_g_src', type=str, default="SU")
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--v_method', type=str, choices=['encoderv', 'reconv'], default='reconv')
parser.add_argument('--steady', type=str, choices=['True', 'False'], default='True')
parser.add_argument('--use_x', type=bool, default=False,
                    help="""whether or not to enroll transcriptom reads for training 
                            (default: False)."""
                    )
parser.add_argument('--refit', type=int, default=1,
                    help="""whether or not refitting veloAE, if False, need to provide
                            a fitted model for velocity projection. (default=1)
                         """
                    )
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay strength (default 0.0)')
parser.add_argument('--ld_adata', type=str, default="projection.h5",
                    help='Path of output low-dimensional adata (projection.h5)')

args = parser.parse_args()

device = torch.device("cuda" + args.device if torch.cuda.is_available() else "cpu")
ODE_type = args.ode_type
seed = args.seed
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

channel_names = 'u, s'
hidden_layers = args.hidden_layer_s
dt = args.dt
if args.use_bias == 'True':
    use_bias = True
else:
    use_bias = False

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
    adata = scv.datasets.dentategyrus()
    cluster_edges = [("OPC", "OL")]
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
    adata = scanpy.read_h5ad('./dataset/scNTseq.h5ad')
    cluster_edges = [("0", "15"), ("15", "30"), ("30", "60"), ("60", "120")]
    k_cluster = "time"
    scv.utils.show_proportions(adata)
    adata.obs['time'] = adata.obs.time.astype('category')
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

elif args.dataset == 'Multi':
    adata = scanpy.read_h5ad('./dataset/scNT-seq.h5ad')
    k_cluster = "time"
    scv.utils.show_proportions(adata)
    adata.obs['time'] = adata.obs.time.astype('category')
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.gene_number)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    chromatin = adata.layers['new']
    chromatin = chromatin.todense()
    tensor_c = torch.DoubleTensor(chromatin).to(device)
    tensor_all = [tensor_c]
    channel_names = 'u, s, c'

exp_metrics = {}
temp_adata = adata
spliced = adata.layers['Ms']
unspliced = adata.layers['Mu']
x = adata.X.toarray()
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

odenet.load_state_dict(torch.load(args.pretrain_odenet, map_location='cpu'))
odenet = odenet.to(device)
odenet.eval()
expr_dict = get_expr(odenet, ifprint=False, transform=None, genes=args.gene_number)
# with open("expr_dict.pkl", "wb") as tf:
#     pickle.dump(expr_dict, tf)
# print_eqs(expr_dict)

# with open('expr_dict.pkl', 'rb') as p:
#     c = pickle.load(p)

with torch.no_grad():
    tensor_s = torch.DoubleTensor(spliced).to(device)
    tensor_u = torch.DoubleTensor(unspliced).to(device)
    # tensor_x = torch.DoubleTensor(adata.X.toarray()).to(device)
    if args.dataset == 'Multi':
        input = torch.cat(
            [tensor_u.unsqueeze(2).unsqueeze(3), tensor_s.unsqueeze(2).unsqueeze(3), tensor_c.unsqueeze(2).unsqueeze(3)],
            dim=-1).to(
            device, dtype=torch.float64)
    else:
        input = torch.cat([tensor_u.unsqueeze(2).unsqueeze(3), tensor_s.unsqueeze(2).unsqueeze(3)], dim=-1).to(
            device, dtype=torch.float64)
    s_embedding = input[:, :, 0, 1]

    output = odenet.step(input, s_embedding).to(dtype=torch.float64)
    v_sym = ((output - input) / dt)[:, :, 0, 1].cpu().numpy()
    # temp_v = v_sym

dsdt = expr_dict[1]
score = []
s1 = sympy.symbols('s')
u1 = sympy.symbols('u')
su0 = sympy.symbols('1')
filter_number = args.filter_number
for idx in dsdt.keys():
    gene = dsdt[idx]
    score_temp = (abs(gene[s1]) + abs(gene[u1])) / sum([abs(gene[k]) for k in gene.keys()])
    score.append(score_temp)
# print(score)
sort_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
sort_id = sort_id[0: filter_number]
sort_id.sort()
spliced = spliced[:, sort_id]
unspliced = unspliced[:, sort_id]
x = x[:, sort_id]
v_sym = v_sym[:, sort_id]
# print(v_sym == temp_v)
adata = new_adata(adata, x, spliced, unspliced, v_sym, g_basis="X")
scv.tl.velocity_graph(adata, vkey='new_velocity')
scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, title="SymVelo", dpi=350, vkey="new_velocity",
                          save='figures/%s/arrow_dml_SymVelo_%s_hl%d_nb_%s_post_%d.png' % (
                              args.dataset, args.psm, args.hidden_layer_s, str(use_bias), filter_number))
scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity", basis='X_umap',
                                 color=k_cluster,
                                 title="SymVelo",
                                 dpi=350,
                                 save='figures/%s/dml_SymVelo_%s_hl%d_nb_%s_post_%d.png' % (
                                     args.dataset, args.psm, args.hidden_layer_s, str(use_bias), filter_number))
scv.tl.velocity_confidence(adata, vkey='new_velocity')
if args.dataset != 'Multi':
    exp_metrics['SymVelo'] = evaluate(adata, cluster_edges, k_cluster, "new_velocity")
