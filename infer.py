import pickle
from pathlib import Path
import matplotlib
import numpy
import torch
import re

from sklearn.decomposition import PCA

from SymNet.model_helper import get_expr
from VeloAe.eval_util import evaluate
from VeloAe.veloproj import *
from VeloAe.model import *
import argparse
from pathlib import Path
from SymNet import polypde
import scvelo as scv
import scanpy
import scipy
import re
from VeloAe.util import estimate_ld_velocity
from VeloAe.veloproj import *
from VeloAe.model import *
from utils import PairsDataset, print_eqs, get_state_change_vector, preprocess_data
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
parser.add_argument('--hidden_layer_s', type=int, default=2, help='Hidden layers of SymNet')
parser.add_argument('--dt', type=float, default=5e-3, help='dt')
parser.add_argument('--use_bias', type=str, choices=['True', 'False'], default='False',
                    help='True: Each linear layers have a bias as usual;'
                         'False: Use a MLP instead of all bias of linear layers of SymNet')
parser.add_argument('--optimizer', type=str, choices=['Adam', 'SDG', 'RMSprop'],
                    default='Adam')  # unfinished only default is work
parser.add_argument('--scheduler', type=str, choices=['Step', 'Lambda', 'Cosine'],
                    default='Step')  # unfinished only default is work
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--figures', type=str)
# data config
parser.add_argument('--dataset', type=str,
                    choices=['pancreas', 'dentategyrus', 'Erythroid_human', 'Erythroid_mouse', 'scEUseq', 'scNTseq',
                             'Multi', 'SHARE', 'Liver', 'GreenLeaf', 'Dyngen_Linear', 'Dyngen_Bifurcation',
                             'Dyngen_Trifurcation', 'Share_new'], default='Multi',
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
parser.add_argument('--lr_decay', type=float, default=0.9)
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

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda" + args.device if torch.cuda.is_available() else "cpu")
    Path(args.figures).mkdir(parents=True, exist_ok=True)
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
    learning_rate = args.lr_s
    cluster_edges = []
    epochs_s = args.epochs_s
    # criterion = nn.MSELoss()
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
        cluster_edges = [("OPC", "OL"), ("nIPC", "Neuroblast"), ("Neuroblast", "Granule immature"),
                         ("Granule immature", "Granule mature"), ("Radial Glia-like", "Astrocytes")]
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
        sel = sel | (adata.obs.celltype == "Early Erythroid").values | (
                adata.obs.celltype == "Mid  Erythroid").values | (
                      adata.obs.celltype == "Late Erythroid").values
        sel = sel | (adata.obs.celltype == "MEMP").values
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
        k_cluster = "celltype"
        cluster_edges = [('TAC-1', 'IRS'), ('TAC-1', 'Medulla'), ('TAC-1', 'Hair Shaft-Cuticle/Cortex')]
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
        adata = scanpy.read_h5ad('./data/greenleaf_multiome/greenleaf_multivelo_0525.h5ad')
        cluster_edges = [("Cyc.", "RG/Astro"), ("Cyc.", "mGPC/OPC"), ("Cyc.", "nIPC/ExN"), ("nIPC/ExN", "ExM"),
                         ("ExM", "ExUp")]
        k_cluster = "cluster"

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
        adata = scv.read('./dataset/synthetic_bifurcation_mile.h5ad')
        k_cluster = 'milestone'
        cluster_edges = [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'E')]
        scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        adata = preprocess_data(adata, filter_on_r2=False)
        args.gene_number = adata.layers['spliced'].shape[1]
        # args.g_rep_dim = adata.layers['spliced'].shape[1]

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
        cluster_edges = [("Cyc.", "RG/Astro"), ("Cyc.", "mGPC/OPC"), ("Cyc.", "nIPC/ExN"), ("nIPC/ExN", "ExM"),
                         ("ExM", "ExUp")]
        #cluster_edges = [("Cyc.", "RG/Astro"), ("RG/Astro", "mGPC/OPC"), ("Cyc.", "nIPC/ExN"), ("nIPC/ExN", "ExM"), ("ExM", "ExUp")]
        k_cluster = "cluster"
        chromatin = adata.layers['Mc']
        chromatin = chromatin.todense()
        tensor_c = torch.DoubleTensor(chromatin).to(device)
        tensor_all = [tensor_c]
        channel_names = 'u, s, c'

    exp_metrics = {}
    temp_adata = adata

    spliced = adata.layers['Ms']
    unspliced = adata.layers['Mu']
    cell_number = adata.layers['Ms'].shape[0]

    tensor_s = torch.DoubleTensor(spliced).to(device)
    tensor_u = torch.DoubleTensor(unspliced).to(device)
    if type(adata.X) == type(adata.layers['Ms']):
        tensor_x = torch.DoubleTensor(adata.X).to(device)
    else:
        tensor_x = torch.DoubleTensor(adata.X.toarray()).to(device)
    # if args.dataset == 'pancreas':
    #     tensor_s = torch.log_(tensor_s + 1)
    #     tensor_u = torch.log_(tensor_u + 1)
    veloae, edge_index = init_model(adata, args, device)
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
    if args.pretrain_model != 'None':
        pretrain_model_temp = torch.load(args.pretrain_model, map_location='cpu')
        veloae.load_state_dict(pretrain_model_temp)
    veloae = veloae.to(device, dtype=torch.float64)
    odenet.load_state_dict(torch.load(args.checkpoint))
    odenet = odenet.to(device)
    odenet.eval()
    veloae.eval()
    with torch.no_grad():
        g_x, r_x = veloae.encoder(tensor_x, True)
        g_s, r_s = veloae.encoder(tensor_s, True)
        g_u, r_u = veloae.encoder(tensor_u, True)
        x = veloae.decoder(r_x, g_x, False)
        u_z = veloae.decoder(r_u, g_u, False)
        s_z = veloae.decoder(r_s, g_s, False)
        if args.psd == 'low':
            v = estimate_ld_velocity(g_s, g_u, device=device)
            input = torch.cat([g_u.unsqueeze(2).unsqueeze(3), g_s.unsqueeze(2).unsqueeze(3)], dim=-1).to(device,
                                                                                                         dtype=torch.float64)
        else:
            if args.dataset == 'Multi':
                g_c, r_c = veloae.encoder(tensor_c, True)
                v = estimate_ld_velocity(g_u, g_c, device=device)
                channel_num = len(tensor_all)
                input = torch.cat([tensor_u.unsqueeze(2).unsqueeze(3), tensor_s.unsqueeze(2).unsqueeze(3)], dim=-1)
                for i in range(channel_num):
                    input = torch.cat([input, tensor_all[i].unsqueeze(2).unsqueeze(3)], dim=-1).to(
                        device, dtype=torch.float64)

            else:
                v = estimate_ld_velocity(g_s, g_u, device=device)
                input = torch.cat([tensor_u.unsqueeze(2).unsqueeze(3), tensor_s.unsqueeze(2).unsqueeze(3)], dim=-1).to(
                    device, dtype=torch.float64)

        s_embedding = input[:, :, 0, 1]
        output = odenet.step(input, s_embedding).to(dtype=torch.float64)
        # v_sym = estimate_ld_velocity(output[:, :, 0, 1], output[:, :, 0, 0], device=device)
        v_sym = ((output - input) / dt)[:, :, 0, 1]
        # kl = torch.nn.KLDivLoss()
        # distance_v = kl(F.log_softmax(v_sym, dim=1), F.softmax(v, dim=-1))
        # p_veloae, p_ode, l1, l2 = get_state_change_vector(edge_index, v, v_sym, s_z, device=device)

        x = x.cpu().numpy()
        v, v_sym = v.cpu().numpy(), v_sym.cpu().numpy()
        s_z, u_z = s_z.cpu().numpy(), u_z.cpu().numpy()
        g_x, g_s, g_u = g_x.cpu().numpy(), g_s.cpu().numpy(), g_u.cpu().numpy()
        if args.pre_model == 'VeloAE':
            print('VeloAE:')
            adata = new_adata(adata, g_x, g_s, g_u, v, g_basis="X", g_rep_dim=args.g_rep_dim)
            scv.tl.velocity_graph(adata, vkey='new_velocity')
            scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, title="VeloAE", dpi=350, vkey="new_velocity",
                                      save='%s/arrow_VeloAE_%s_hl%d_nb_%s.png' % (
                                          args.figures, args.psm, args.hidden_layer_s, str(use_bias)))
            if args.dataset == 'Multi':
                scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity", basis='umap',
                                                 color=k_cluster,
                                                 title="VeloAE",
                                                 dpi=350,
                                                 save='%s/VeloAE_%s_hl%d_nb_%s.png' % (
                                                     args.figures, args.psm, args.hidden_layer_s, str(use_bias)))
            else:
                scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity", basis='X_umap',
                                                 color=k_cluster,
                                                 title="VeloAE",
                                                 dpi=350,
                                                 save='%s/VeloAE_%s_hl%d_nb_%s.png' % (
                                                     args.figures, args.psm, args.hidden_layer_s, str(use_bias)))
            scv.tl.velocity_confidence(adata, vkey='new_velocity')
            if len(cluster_edges) != 0:
                exp_metrics['VeloAE'] = evaluate(adata, cluster_edges, k_cluster, "new_velocity")
        elif args.pre_model == 'scVelo':
            print('scVelo:')
            scv.tl.velocity(adata, vkey='stc_velocity', mode="stochastic")
            scv.tl.velocity_graph(adata, vkey='stc_velocity')
            scv.tl.velocity_confidence(adata, vkey='stc_velocity')
            scv.pl.velocity_embedding_stream(adata, vkey="stc_velocity", basis='umap', color=k_cluster, dpi=350,
                                             title='ScVelo Stochastic Mode', save='figures/%s/scVelo.png' % (
                    args.dataset))
            if len(cluster_edges) != 0:
                exp_metrics["stc_mode"] = evaluate(adata, cluster_edges, k_cluster, "stc_velocity")
        else:
            print('other:')

            v = numpy.load(args.pre_model)
            adata.layers['new_velocity'] = v
            scv.tl.velocity_graph(adata, vkey='new_velocity')
            scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, title="Other", dpi=350,
                                      vkey="new_velocity",
                                      save='figures/%s/arrow_Other_%s_hl%d_nb_%s.png' % (
                                          args.dataset, args.psm, args.hidden_layer_s, str(use_bias)))
            if args.dataset == 'Multi':
                scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity",
                                                 basis='umap',
                                                 color=k_cluster,
                                                 title="Other",
                                                 dpi=350,
                                                 save='figures/%s/Other_%s_hl%d_nb_%s.png' % (
                                                     args.dataset, args.psm, args.hidden_layer_s, str(use_bias)))
            else:
                scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity",
                                                 basis='X_umap',
                                                 color=k_cluster,
                                                 title="Other",
                                                 dpi=350,
                                                 save='figures/%s/Other_%s_hl%d_nb_%s.png' % (
                                                     args.dataset, args.psm, args.hidden_layer_s, str(use_bias)))
            scv.tl.velocity_confidence(adata, vkey='new_velocity')
            if len(cluster_edges) != 0:
                exp_metrics['Other'] = evaluate(adata, cluster_edges, k_cluster, "new_velocity")

        print('SymVelo')
        if args.psd == 'low':
            adata = new_adata(temp_adata, g_x, g_s, g_u, v_sym, g_basis="X", g_rep_dim=args.g_rep_dim)
        else:
            # adata = new_adata(temp_adata, x, s_z, u_z, v_sym, g_basis="X")
            adata = new_adata(temp_adata, tensor_x.cpu().numpy(), tensor_s.cpu().numpy(), tensor_u.cpu().numpy(), v_sym,
                              g_basis="X", g_rep_dim=args.g_rep_dim)
        scv.tl.velocity_graph(adata, vkey='new_velocity')
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, title="SymVelo", dpi=350, vkey="new_velocity",
                                  save='%s/arrow_dml_SymVelo_%s_hl%d_nb_%s.png' % (
                                      args.figures, args.psm, args.hidden_layer_s, str(use_bias)))

        if args.dataset == 'Multi':
            scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity", basis='umap',
                                             color=k_cluster,
                                             title="SymVelo",
                                             dpi=350,
                                             save='%s/dml_SymVelo_%s_hl%d_nb_%s.png' % (
                                                 args.figures, args.psm, args.hidden_layer_s, str(use_bias)))
        else:
            scv.pl.velocity_embedding_stream(adata, legend_loc='right_margin', vkey="new_velocity", basis='X_umap',
                                             color=k_cluster,
                                             title="SymVelo",
                                             dpi=350,
                                             save='%s/dml_SymVelo_%s_hl%d_nb_%s.png' % (
                                                 args.figures, args.psm, args.hidden_layer_s, str(use_bias)))
        scv.tl.velocity_confidence(adata, vkey='new_velocity')
        exp_metrics = {}
        exp_metrics['SymVelo'] = evaluate(adata, cluster_edges, k_cluster, "new_velocity")
        # if len(cluster_edges) != 0:
        #     exp_metrics['SymVelo'] = evaluate(adata, cluster_edges, k_cluster, "new_velocity")
        # # print(v_sym)
        # if args.psd == 'high':
        #     expr_dict = get_expr(odenet, ifprint=False, genes=args.gene_number)
        # else:
        #     expr_dict = get_expr(odenet, ifprint=False, genes=args.z_dim)
        # with open("./checkpoint/%s/expr_dict.pkl" % args.dataset, "wb") as tf:
        #     pickle.dump(expr_dict, tf)

        #
        # print(distance_v)
        # print(l1+l2)
        # print(p_veloae)
        # print(p_ode)

        #
        # torch.save(torch.from_numpy(v_sym), "./checkpoint/%s/v_sym.pt" % args.dataset)
        # print("Save v_sym")
        # p_sym, p_velo, _, _ = get_state_change_vector(edge_index, torch.from_numpy(v_sym).to(device), torch.from_numpy(v).to(device), torch.from_numpy(g_s).to(device), tensor_s.to(device), device=device)
        # torch.save(p_sym.cpu(), "./checkpoint/%s/p_sym.pt" % args.dataset)
        # torch.save(p_velo.cpu(), "./checkpoint/%s/p_velo.pt" % args.dataset)
        # print("Save p_sym & p_velo")
        # torch.save(edge_index, "./checkpoint/%s/edge_index.pt" % args.dataset)
        # dudt = ((output - input) / dt)[:, :, 0, 0]
        # if args.dataset == 'Multi':
        #     dcdt = ((output - input) / dt)[:, :, 0, 2]
        # torch.save(dudt.cpu(), "./checkpoint/%s/dudt.pt" % args.dataset)
        # torch.save(torch.from_numpy(g_u), "./checkpoint/%s/gu.pt" % args.dataset)
        # torch.save(torch.from_numpy(g_s), "./checkpoint/%s/gs.pt" % args.dataset)
        # torch.save(torch.from_numpy(g_x), "./checkpoint/%s/gx.pt" % args.dataset)
        # torch.save(torch.from_numpy(v), "./checkpoint/%s/v_velo.pt" % args.dataset)
