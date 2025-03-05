from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
#from DDPM.ddpm.utils import extract
from functools import partial
import math
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics._classification import cohen_kappa_score, accuracy_score
from munkres import Munkres


class HSIDataset(Dataset):
    def __init__(self, img_path, gt_path, im, nb_comps, NEIGHBORING_SIZE):
        nb_comps = nb_comps    # 目标通道数
        p = Processor()
        img, gt = p.prepare_data(img_path, gt_path)
        self.img = img
        if im == 'WHU_Hi_LongKou':
            img, gt = img[110:220,208:275, :], gt[110:220,208:275]  # 200*100 * 103
            NEIGHBORING_SIZE = NEIGHBORING_SIZE

        if im == 'WHU_Hi_HongHu':
            img, gt = img[800:900,90:170, :], gt[800:900,90:170]  # 200*100 * 103
            NEIGHBORING_SIZE = NEIGHBORING_SIZE
        
        if im == 'PaviaU':
            img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]  # 200*100 * 103
            NEIGHBORING_SIZE = NEIGHBORING_SIZE

        if im == 'Indian_pines_corrected':
            img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]     # 85*70 * 200
            NEIGHBORING_SIZE = NEIGHBORING_SIZE

        if im == 'SalinasA_corrected':       # 83*86 * 204
            NEIGHBORING_SIZE = NEIGHBORING_SIZE

        self.gt = gt
        n_row, n_column, n_band = img.shape

        img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)   # reshape拉长归一化后再reshape回原形状

        # perform PCA
        pca = PCA(n_components=nb_comps)      # 通道降为5
        img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))

        #print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
        print('pca shape:', img.shape)
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (NEIGHBORING_SIZE, NEIGHBORING_SIZE))  # x_patch=(n_samples, n_width, n_height, n_band)
        #20000, 13, 13, 5
        print("cluster: ", np.unique(y_))  # 排列后输出  标签值几类
        # perform ZCA whitening
        # x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
        # x_patches, _, _ = p_Cora.zca_whitening(x_patches, epsilon=10.)
        self.x = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
        #print(self.x)
        
        #noise = np.random.normal(0,0.1,(NEIGHBORING_SIZE, NEIGHBORING_SIZE,nb_comps))
        #self.x += noise
        
        #print('img shape:', img.shape)
        print('img_patches_nonzero:', self.x.shape)
        #n_samples, n_width, n_height, n_band = self.x.shape

        self.y = p.standardize_label(y_)
        print(np.unique(self.y))
 
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return self.x, self.y        # 标准化后的数据和标签

    def get_nclusters(self):
        return np.unique(self.y).shape[0]  # 获得类别数

    def get_nsamples(self):
        return self.x.shape[0]  # 获得标签数

    def get_y_hat(self):

        return self.gt, self.y      # 83 86      标签归一化label

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=1, )
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=1, )
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, )

        self.bn1 = nn.BatchNorm2d(num_features = 24)
        self.bn2 = nn.BatchNorm2d(num_features = 24)
        self.bn3 = nn.BatchNorm2d(num_features = 32)
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.rl(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.rl(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.rl(x3)

        return x3


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model."""

    def __init__(
            self,
            model,
            img_size,
            img_channels,
            betas,
    ):
        super(GaussianDiffusion, self).__init__()

        self.model = model
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels

        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))  #
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))  #
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, t):

        return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    def perturb_x(self, x, t, noise):
        # 计算t时刻的噪音

        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t):
        t1 = torch.full((x.shape[0],), t)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t1, noise)
        estimated_noise = self.model(perturbed_x, t1)

        # t1 = torch.full((x.shape[0],), 8)
        # # t1 = torch.full((x.shape[0], ), 12)
        # noise1 = torch.randn_like(x, device=x.device)
        # perturbed_xt = self.perturb_x(x, t1, noise1)
        # estimated_noise1 = self.model(perturbed_xt, t1)

        loss = F.mse_loss(estimated_noise, noise)

        return loss, estimated_noise

    def forward(self, x, t):
        #b, c, h, w = x.shape
        #device = x.device

        # if h != self.img_size[0]:
        #     raise ValueError("image height does not match diffusion parameters")
        # if w != self.img_size[0]:
        #     raise ValueError("image width does not match diffusion parameters")

        #t = torch.randint(0, self.num_timesteps, (b,), device=device)
        # t1 = torch.full((x.shape[0],), t)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise, feature = self.model(perturbed_x, t)

        # loss = F.mse_loss(estimated_noise, noise)

        return noise, estimated_noise, feature

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=32+channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        #self.convt1 = nn.ConvTranspose2d(in_channels=32+32, out_channels=32, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.convt2 = nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.convt3 = nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1) ,padding=1,)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=channels, kernel_size=(1, 1), stride=(1, 1),)

        self.bn1 = nn.BatchNorm2d(num_features = 32)
        self.bn2 = nn.BatchNorm2d(num_features = 24)
        self.bn3 = nn.BatchNorm2d(num_features = 24)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.rl(x)

        x = self.convt2(x)
        x = self.bn2(x)
        x = self.rl(x)

        x = self.convt3(x)
        x = self.bn3(x)
        x = self.rl(x)

        x = self.conv4(x)
        return x

class DOT(nn.Module):
    def __init__(self, nb_comps, nsamples, T, NEIGHBORING_SIZE, beta):
        super(DOT, self).__init__()
        self.Encoder = Encoder(nb_comps)
        self.Decoder = Decoder(nb_comps)
        self.Extractor = Extractor(nb_comps, T, NEIGHBORING_SIZE)
        self.Diffusion = GaussianDiffusion(self.Extractor, (NEIGHBORING_SIZE,NEIGHBORING_SIZE), nb_comps, beta)
        self.mse = torch.nn.MSELoss()
        self.C = torch.nn.Parameter(5.0e-5 * torch.ones(nsamples, nsamples))
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
 
    def SelfExpress(self, h):
        h_temp = h.permute(0, 2, 3, 1)    # 20000 13 13 32
        z = h_temp.flatten(1)   # 从第一维展开  20000 5408
        n = z.shape[0]
        z_hat = torch.matmul(self.C, z)   # 20000*20000   20000*5408
        h_temp = z_hat.view(h_temp.shape)   # 改变张量维度 20000 13 13 32
        h_hat = h_temp.permute(0, 3, 1, 2)     # 20000 32 13 13
        return z, z_hat, self.C, h_hat

    def forward(self, x, t, device):
        t = torch.full((x.shape[0],), t).to(device)
        #e = torch.randn_like(x)
        h1 = self.Encoder(x)
        eps, eps_p, feature = self.Diffusion(x, t)
        #h = torch.cat((h1, e), 1)
        h = torch.cat((h1, feature), 1)
        #h = torch.cat((feature, eps_p), 1)
        z, z_hat, C, h_hat = self.SelfExpress(h)
        x_hat = self.Decoder(h_hat)
        return z, z_hat, x_hat, C, eps, eps_p

    def loss_wass(self, z, z_hat, x, x_hat, C, eps, eps_p, REG_LATENT, WEIGHT_DECAY,):  # 100  0.1
        #print(REG_LATENT, WEIGHT_DECAY)
        # loss = self.mse(z, z_hat) + self.mse(x, x_hat)
        m, _, _, _ = x.shape
        #a = torch.ones(m) / m        # 20000
        x = x.reshape([m, -1]).cpu()    # 20000 845
        x_hat = x_hat.reshape([m, -1]).cpu()  # 20000 845

        eps = eps.reshape([m, -1]).cpu()
        eps_p = eps_p.reshape([m, -1]).cpu()
        # M = ot.dist(x, x_hat)
        # M = M / M.max()
        #
        # ls1 = ot.emd2(a, a, M)
        ls1 = self.mse(x, x_hat)
        ls2 = self.mse(z_hat, z)
        ls3 = torch.norm(C, p = 2)
        ls4 = self.mse(eps, eps_p)

        # print(ls1.data.cpu().numpy(), ls2.data.cpu().numpy(), ls3.data.cpu().numpy())
        loss = ls1 + REG_LATENT * ls2 + WEIGHT_DECAY * ls3 + ls4
        # --------------------------------------------------
        return loss

class SpectralClustering():
    def __int__(self):
        pass

    def predict(self, Coef, n_clusters, alpha=0.25):    #自表达  簇数
        Coef = self.thrC(Coef, alpha)
        y_pre, C = self.post_proC(Coef, n_clusters, 8, 18)
        #np.savez('./models/Affinity.npz', coef=C)
        #np.savez('./models/y_pre.npz', y_pre=y_pre)
        # missrate_x = self.err_rate(y, y_x)
        # acc = 1 - missrate_x
        return y_pre, C

    def thrC(self, C, ro):      # C 样本数*样本数
        if ro < 1:
            N = C.shape[1]     # 样本数
            Cp = np.zeros((N, N))

            S = np.abs(np.sort(-np.abs(C), axis=0))   # 把每列从大到小向下排列
            Ind = np.argsort(-np.abs(C), axis=0)     # 每列从大到小排列的下标

            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)   # 所有行, 第i列和
                stop = False
                csum = 0
                t = 0  # 行号
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:     # 0.3 * 第i列
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        # C:系数矩阵，             K:聚类数，              d:每个子空间的维数 8                alpha: 18
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        #grp = spectral.fit_predict(L) + 1
        grp = spectral.fit_predict(L)
        return grp, L

    def cluster_accuracy(self, y_true, y_pre):    #1-6  0-5
        Label1 = np.unique(y_true)    # 去除其中重复的元素,并按元素由小到大返回一个新的无元素重复的列表
        #print(Label1)
        nClass1 = len(Label1)   # 原始类别数

        Label2 = np.unique(y_pre)
        #print(Label2)
        nClass2 = len(Label2)   # 预测类别数

        nClass = np.maximum(nClass1, nClass2)    # 最大类别
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)   # 真实第i类 与 预测第j类 的相同像素个数
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.class_acc(y_true, y_best)
        #print(np.unique(y_best))
        return acc, nmi, kappa, ca, y_best

    def class_acc(self, y_true, y_pre):
        """
        calculate each classes's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca

def draw(y, y_pre):
    y_pre += 1
    y_non = np.zeros(y.shape)
    sets = np.nonzero(y)
    assert y_pre.shape == sets[0].shape
    y_non[sets] = y_pre

    return y_non

def scale(C):
    mx = np.nanmax(C)
    mn = np.nanmin(C)
    t = (C - mn) / (mx - mn)
    return t

def extract(a, t, x_shape):
    t.to(torch.device('cuda:0'))
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


class Extractor(nn.Module):
    def __init__(self, channels, T, neighbor):
        super(Extractor, self).__init__()

        self.time_embedding1 = nn.Embedding(T, neighbor * neighbor)
        self.time_embedding2 = nn.Embedding(T, neighbor * neighbor)
        self.time_embedding3 = nn.Embedding(T, neighbor * neighbor)

        self.Seq1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            # (16, 28, 28)
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),  # (16, 14, 14)
        )
        self.Seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32, 14, 14)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # nn.MaxPool2d(2),  # (32, 7, 7)
        )
        self.Seq3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32, 14, 14)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # nn.MaxPool2d(2),  # (32, 7, 7)
        )
        self.conv = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=1, stride=1)


    def forward(self, x0, t):

        emb1 = self.time_embedding1(t)
        x1 = x0 + torch.reshape(emb1, (x0.shape[0], 1, x0.shape[2], x0.shape[3]))
        x2 = self.Seq1(x1)

        emb2 = self.time_embedding2(t)
        x3 = x2 + torch.reshape(emb2, (x0.shape[0], 1, x0.shape[2], x0.shape[3]))
        x4 = self.Seq2(x3)

        emb3 = self.time_embedding3(t)
        x5 = x4 + torch.reshape(emb3, (x0.shape[0], 1, x0.shape[2], x0.shape[3]))
        x6 = self.Seq3(x5)

        x7 = self.conv(x6)

        #print(feature.shape)


        return x7, x7       # 5, 32

