# SymVelo

SymVelo is a a dual-path framework to estimate RNA velocity, which first trains two branches of neural networks for high- and low-dimensional RNA velocities, respectively. The framework then aligns them via mutual learning. It successfully inherits the robustness from representation learning via the low-dimensional branch while preserving biological interpretability through the high-dimensional one. Furthermore, mutual learning covers all cells for each latent dimension, which provides inter-gene information on the supervision of representation learning.

The framework is as follows:
<div align=center><img width="900" src="framework.png"></div>


SymVelo consists of three modules, including the temporal difference module, the pre-trained representation learning module and the mutual learning module. 
1. The temporal difference module aims to estimate the continuous high-dimensional RNA velocity via Neural ODE in a bottom-up manner. The main component of this module is a tailored symbolic network [SymNet](https://arxiv.org/abs/1710.09668), which models the transcriptional gene dynamics via the generalized kinetic model. 
2. The representation learning branch aims to learn a low-dimensional representation of RNA velocity to avoid the sparsity and noise of the raw counts, and it has been shown effective to reveal robust cell transitions in [VeloAE](https://www.biorxiv.org/content/10.1101/2021.03.19.436127v1). 
3. We adopt [deep mutual learning (DML)](https://ec.europa.eu/research-and-innovation/en/statistics/policy-support-facility/mutual-learning) to further align the transition probabilities obtained from the two branches.



## Installation

It is recommended to create an environment with [requirement.txt](requirement.txt), Otherwise you have to be careful when installing packages like ``scvelo`` and ``scipy``. If the code does not work, please check the following versions. For training SymVelo model, the GPU should have 11GB (GTX 1080ti). 

```
torch                 1.9.0
torch-geometric       2.0.3
scanpy                1.8.2
h5py                  3.6.0
scanpy                1.8.2
sympy                 1.10.1
veloAE                0.0.2 
anndata               0.7.8
numba                 0.55.1
numpy                 1.20.0
```

## Usage

### VeloAE Pre-training

We modified VeloAE to better coordinate with SymNet, the modified VeloAE is in ``VeloAE`` folder. Take the [pancreas dataset](model-pancreas.ipynb) as an example, we can get a VeloAE pre-trained model, the velocity graph and metrics (ICVCoh and CBDir) via scvelo and VeloAe :

```
ipython model-pancreas.ipynb
```

For other datasets, we only need to adjust the parameters and the data preprocessing method (if any) in the ``.ipynb`` files.
```
# parameters

'--data-dir', './dataset/endocrinogenesis_day15.5.h5ad', # the path of dataset
'--model-name', './pretrain_model/pancreas_test.cpt', # the path to save pretrain model


# preprocessing (if any)

adata = scanpy.read_h5ad(args.data_dir)
# your method
scv.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scv.utils.show_proportions(adata)
scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=args.n_raw_gene)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
```

### SymNet Training
We redesign the network and coefficient calculation logic so that SymNet can calculate multiple genes at the same time, the modified VeloAE is in ``SymNet`` folder.
We train SymNet and pre-trained VeloAE via DML method in [main.py](main.py). Each important parameters are commented in their respective help, and only a few parameters need to be changed while trainning or velocity calculating, like:

```
pretrain_model, use_bias, epochs_s, lr_s, lr_v, psm, dt, dataset, gumbsoft_tau, psd, frozen, veloae_coef, hidden_layer_s, checkpoint.
```

For scNT-seq dataset, the command is like (more commands and results can be found in [results.xlsx](results.xlsx) ):

```
python main.py --pretrain_model ./pretrain_model/scNT_model.cpt --use_bias True --epochs_s 300 --lr_v 1e-5 --lr_s 1e-3 --psm random --dt 1  --dataset scNTseq --gumbsoft_tau 5 --psd high --frozen False
```

From line 82 to line 254, we will process the parameters:
+ Importantly, if experimenting on other datasets, we need to add name of dataset to the choices of parameter ``dataset``(line 40) and imitate other datasets to complete data reading and preprocessing. If new datasets are Multi-omics datasets, we recommend modifying line 169-188 directly, We also show in the comments that how to rewrite in more omics. (If you have to create a new dataset branch, remember to change all code where ``if args.dataset == 'Multi':`` appear.)
+ We define VeloAE and SymNet model from 197 to line 250, and adjust the model initialization method and loss function according to the magnitude of the loss function on different tasks in training.

From line 266 to line 367, we frozen VeloAE and train SymNet alone; From line 367 to line 547, we train VeloAE and SymNet together, and adopt DML to further align the transition probabilities obtained from VeloAE and SymNet:
+ First of all, we sample pairs for SymNet training by the velocity calculated by the VeloAE pre-trained model. For cell $i$, cell’s expression state $x \in \mathbb{R}^{M\cdot d}$, the neighbor cells $j \in N(i)$, velocity $v$ and spliced RNAs in latent space $x^z$, we sample $(i, j)$ as a pair if the direction of truthful cellular state change from $i$ to $j$ is close to velocity $v_i$. We set three sampling methods(details in [utils.py](utils.py)): 
    1. random: 90% probability to select $j = \mathop{\arg \max} \  \cos \langle v_i, x_j^z - x_i^z \rangle $ and 10% probability to sample a random neighbor cell as $c_j$, which result in pair $(i, j)$.
    2. all: pair each cell with all its neighbors.
    3. randomv2: similar to random method, but sample 5 pairs with top-5 closest cells. 

    Then, we get pairs for each cell and concatenate cell state of each omics to obtain the truthful implied future cell state $x^+$. We update the pairs for several epoch(s) of training(now every epoch) to ensure that the two models can learn from each other and explore the correct velocities.

+ Calculate velocities and loss funtions from SymNet $V_h$ and VeloAE $V_l$:
    1. For cell’s expression state $x$ and the time scale $dt$, $$V_h = \frac{SymNet(x) - x}{dt}.$$
    2. Step loss function is MSE, $$L_{step} = \sum_{(x,x^+)} \left \| \frac{SymNet(x) - x}{dt}\right \|_2^2.$$
    3. Sparse loss function is to constrain the sparsity of the network parameters
    4. VeloAE velocity and loss funcion can refer to its article.

+ Transition probability matrix
    1. Given cell’s expression state $x$ and $V_h$, we can obtain the transition probability matrix $P^s$(details in ``get_state_change_vector`` function of [utils.py](utils.py)).  
    2. For cell $i$ and its neighbor cell $j \in N(i)$, $$P_{i,j}^s = \frac{1}{z_{i}} \exp \left( \cos \langle v_i, x_j - x_i \rangle\right), $$ with $z_i = \sum_{j=1}^N \exp \left( \cos \langle v_i, x_j - x_i \rangle\right)$.  
    3. Similarly, we can calculate the transition probability matrix of VeloAE $P_v$ via the cell’s expression state in latent space $x^z$ via VeloAE's encoder and $V_l$ (both $V_l$ and $x^z$ are low-dimensional).  
    4. We then adopt the symmetric Jensen-Shannon Divergence loss as the mimicry loss.
    5. We have made some optimizations in this part, but the calculation and backpropagation of the divergence loss is still a time bottleneck.

+ Count losses, get and save the coefficients of SymNet, save checkpoint and so on.

### SymNet Results

We can get the velocity graphs and the metrics of SymNet and VeloAE by checkpoint we save in [infer.py](infer.py). Some parameters should be the same as when training.  
For scNT-seq dataset, the command corresponding to the above is (more commands and results can be found in [results.xlsx](results.xlsx) ):

```
python infer.py --pretrain_model ./checkpoint/scNTseq/veloae_randomhigh_hl2_nb_True.pth --checkpoint ./checkpoint/scNTseq/odenet_randomhigh_hl2_nb_True.pth --psm random --frozen False --dt 1 --use_bias True --dataset scNTseq --gumbsoft_tau 5 --psd high
```

### A post-processing method for filtering genes

In [filter_gene.py](filter_gene.py), we describe how to get and count Symnet's coefficients. 
+ From line 206 to line 215, we get the coefficients of SymNet which is time-consuming, we can also read the file ``expr_dict.pkl`` to get it if save during training (read and write commands are in comments).
+ From line 237 to line 252, we selecte the $k$ genes with the largest proportion of linear term and constant term coefficients. We hope that in this way we can remove genes that do not fit the hypothesis (like Murk), but it does not work well. You can design other rules in this part.

2022/9/9

main.py
```
parser.add_argument('--batch_size', type=int, default=2000)  # SymNet batch size

parser.add_argument('--pre_model', type=str, choices=['scVelo', 'VeloAE'], default='VeloAE')  # when args.frozen == True, we can choose v from scVelo or VeloAE
```