import sys
import os
from dot_diff import HSIDataset,  DOT, SpectralClustering, draw, generate_linear_schedule, scale   #Encoder, Decoder, GaussianDiffusion, Extractor,
import torch
import torch.backends.cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
import gc
import time
matplotlib.use("Agg")


# if __name__ == '__main__':
# load img and gt
#for NEIGHBORING_SIZE in [3,5,7,9,11,13,15,17]:
#for NEIGHBORING_SIZE in [11,13,15,17]:
#for REG_LATENT in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:                      #lameda
#for WEIGHT_DECAY in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:                   #gamma
#for SEED in [74]:          #InP
#for SEED in [160]:
#for SEED in [91]:   #PaU  111111
#for nb_comps in range(7,14,1):    
#for nb_comps in [13]:
#for t in range(5,15,1):
#for t in range(500,508,1):
#for t in range(990,999,1):    
for SEED in range(1,150,1):     
    print("\n")
    #SEED = 10
    data_root = 'HSI_datasets'

    #im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    #im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    #im_, gt_ = 'PaviaU', 'PaviaU_gt'
    #im_, gt_ = 'WHU_Hi_HongHu','WHU_Hi_HongHu_gt'
    im_, gt_ = 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt'

    img_path = os.path.join(data_root, (im_ + '.mat'))
    gt_path = os.path.join(data_root, (gt_ + '.mat'))
    print("dataset : ", im_)

    # for nb_comps in range(2, 31, 1):
    # for size in range(5, 31, 2):
    T=1000
    #NEIGHBORING_SIZE = 13
    EPOCH = 100

    LEARNING_RATE = 0.0002
    REG_LAP = 0.001  # beta
    REG_LATENT = 100.  # alpha
    WEIGHT_DECAY = 0.001  # lambda
    #SEED = None  # random seed
    nb_comps = 6
    VERBOSE_TIME = 50
    BATCH_SIZE = 1

    #t=12
    schedule_low=1e-4
    schedule_high=0.02

    beta = generate_linear_schedule(T, schedule_low * 1000 / T, schedule_high * 1000 / T,)

    #for SEED in range(1, 201, 10):
        #print("SEED = ", SEED)
    if im_ == 'WHU_Hi_LongKou':
        t=11
        SEED = 175
        nb_comps = 5
        EPOCH = 100
        NEIGHBORING_SIZE=9
        LEARNING_RATE = 0.0005
        #LEARNING_RATE = 0.0012
        REG_LATENT = 100.  # lambda
        WEIGHT_DECAY = 0.01  # rho
    
    if im_ == 'WHU_Hi_HongHu':
        t=11
        #t=9
        SEED = 61
        nb_comps = 5
        EPOCH = 100
        NEIGHBORING_SIZE=13
        LEARNING_RATE = 0.0008
        #LEARNING_RATE = 0.0011
        REG_LATENT = 100.  # lambda
        WEIGHT_DECAY = 0.01  # rho
    
    if im_ == 'PaviaU':
        #for SEED in [31,37,56,74,78,91]:   #PaU  111111
        t=9
        #SEED=31
        EPOCH = 100
        nb_comps = 10
        #LEARNING_RATE = 0.0001      #diff+encoder
        #LEARNING_RATE = 0.00006        #diff
        NEIGHBORING_SIZE=13
        LEARNING_RATE = 0.0008
        REG_LATENT = 100.   # lambda
        WEIGHT_DECAY = 0.1    # rho

    if im_ == 'Indian_pines_corrected':
        t=9
        EPOCH = 100
        SEED=160
        #SEED = 1
        NEIGHBORING_SIZE=13
        #LEARNING_RATE = 0.008
        LEARNING_RATE = 0.0002
        nb_comps=6
        t=12
        REG_LATENT = 100.  # lambda
        WEIGHT_DECAY = 0.01  # rho

    if im_ == 'SalinasA_corrected':
        t=5
        SEED=20
        nb_comps=5
        EPOCH = 100
        NEIGHBORING_SIZE=9
        #LEARNING_RATE = 0.0002
        LEARNING_RATE = 0.0005
        REG_LATENT = 100.  # lambda
        WEIGHT_DECAY = 0.01  # rho
    EPOCH=100
    print("seed:", SEED, ";   netghbor size:", NEIGHBORING_SIZE, ";   nb_comps:", nb_comps, ";   time step : ", t)
    print("epoch : ", EPOCH, "learning rate : ", LEARNING_RATE)
    print("lameda : ", REG_LATENT, "gamma : ", WEIGHT_DECAY)
    device = torch.device('cuda:1')
    #device = torch.device('cpu')

    # 随机种子
    if SEED is not None:
        torch.manual_seed(SEED)
        if device == torch.device('cuda:1'):
            torch.backends.cudnn.deterministic = True  # 设置随机种子
            torch.backends.cudnn.benchmark = False

    train_data = HSIDataset(img_path, gt_path, im_, nb_comps, NEIGHBORING_SIZE)

    #print(train_data)
    nclusters = train_data.get_nclusters()   # 类别数
    nsamples = train_data.get_nsamples()     # 样本数
    y_hat, label = train_data.get_y_hat()
    #print(np.unique(label))
    print("num clusters: %s" % (nclusters))

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,   # 1
        shuffle=False,
        pin_memory=torch.cuda.is_available())

    # Encoder = Encoder(nb_comps)      # channel 5
    #
    # Decoder = Decoder(nb_comps)
    # Extractor = Extractor(nb_comps, T, NEIGHBORING_SIZE)
    # Diffusion = GaussianDiffusion(Extractor, (NEIGHBORING_SIZE,NEIGHBORING_SIZE), nb_comps, beta)

    model = DOT(nb_comps, nsamples, T, NEIGHBORING_SIZE, beta).to(device)

    sc = SpectralClustering()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print("model parameters : ", num_s)
    loss_func = model.loss_wass

    '''
    for data in train_loader:
        x, y = data
        _, _, m, n = x.shape
    a1 = torch.ones(m) / m
    a1 = a1.clone().detach().requires_grad_(True)
    lr=1e-2
    '''
    start = time.localtime(time.time())
    start = time.mktime(start)
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            # print(x.shape, y.shape)
            y = y.clone().detach().squeeze().numpy()
            #print(y.shape)
            x = torch.squeeze(x)
            x = x.permute(0, 3, 1, 2)   # 20000 5 13 13

            # x = (x * w).type(torch.int8) + x
            # print('__________')
            # print(x.shape)
            # print(type(x))

            b, c, m, n = x.shape
            x = x.float().to(device)

            z, z_hat, x_hat, C_1, eps, eps_p = model(x, t, device)
            loss = loss_func(z, z_hat, x, x_hat, C_1, eps, eps_p, REG_LATENT, WEIGHT_DECAY, )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #colors = ['black', 'red', 'darkorange', 'limegreen', 'yellow', 'blue', 'c', 'purple', 'slategray']
            #colors = ['black', 'red', 'darkorange', 'limegreen', 'yellow', 'blue', 'c', 'purple', 'slategray', 'green',
            #          'blueviolet', 'goldenrod', 'bisque', 'plum', 'mediumvioletred', 'gray', 'burlywood']
            #cmap = matplotlib.colors.ListedColormap(colors)
            #settime = time.time()
            '''
            if epoch == 0:
                      # save label
                label1 = draw(y_hat, y)
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.imshow(label1, cmap=cmap)
                # plt.imshow(y_pic, cmap='turbo')  # spectral
                plt.savefig("%s_y_gt.svg" % (im_,), format='svg', bbox_inches='tight')
                plt.close('all')

            '''
            if (epoch + 1) % VERBOSE_TIME == 0:      # 5
                ro = 0.3

                C_2 = C_1.clone().detach().cpu().numpy()

                #print(C_2.min(), C_2.max())
                #torch.save(model.state_dict(), 'D:\\Desktop\\models\\%s_net_epoch_%s.pkl' % (im_, epoch))
                # 保存模型
                #np.save('D:\\Desktop\\models\\%s_C_epoch_%s_%s.npy' % (im_, epoch+1, settime), C_2)
                '''
                plt.figure()
                # 保存亲和矩阵
                plt.imshow(C_2)  # spectral
                plt.savefig("models/%s_C_epoch_%s_%s.svg" % (im_, epoch+1, settime), format='svg', bbox_inches='tight')
                # 保存谱图
                '''
                y_pre, affinity = sc.predict(C_2, nclusters, ro)    # 亲和矩阵 簇数 0.3

                '''
                affinity = np.sqrt(np.sqrt(affinity))
                # affinity = scale(affinity)

                x_major_locator = MultipleLocator(1000)
                y_major_locator = MultipleLocator(1000)

                plt.figure()
                ax = plt.gca()
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)

                plt.imshow(affinity, cmap='jet')

                plt.colorbar()
                plt.savefig("models/%s_affinity_%s_%s.svg" % (im_, epoch+1, settime), format='svg', bbox_inches='tight')
                '''
                acc, nmi, kappa, ca, spe_pre = sc.cluster_accuracy(y, y_pre)
                #print(spe_pre.shape)
                '''
                y_pic = draw(y_hat, spe_pre)
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.imshow(y_pic, cmap=cmap)

                #plt.imshow(y_pic, cmap='turbo')  # spectral
                plt.savefig("models/%s_pre_%s_%s.svg" % (im_, epoch+1, settime), format='svg', bbox_inches='tight')
                '''
                # plt.figure()
                # plt.imshow(spe_pre)
                # plt.savefig("D:\\Desktop\\models\\%s_pre_%s.pdf" % (im_, epoch), format='pdf')
                # print('Epoch: %s, loss: %s' % (epoch, loss.data.cpu().numpy()))
                # print('Epoch: %s, loss: %s, acc:%s' % (epoch, loss.data.cpu().numpy(), (acc, nmi, kappa, ca)))
                print('Epoch: %s\t loss: %s\t ( acc: %s, nmi: %s, ka: %s) \t %s' % (epoch + 1, loss.data.cpu().numpy(), acc, nmi, kappa, ca))
                #plt.close('all')

    #junk = gc.collect()
    finish = time.localtime(time.time())
    finish = time.mktime(finish)
    print("running time:", finish-start)
