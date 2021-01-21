# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        """
        Initialize a new network
        在使用embedding layer的时候切记三步走，一是建立索引字典，二是根据索引字典映射原始数据。三是根据索引字典得到feature_size之后才建立embedding layer。

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.float

        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """

        # self.fm_first_order_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])

        # 对连续特征使用 全连接神经网络层 dense layer -- pytorch 是 nn.Linear
        fm_first_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])

        # 对离散特征使用 嵌入层
        fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])

        # 组合前两组
        self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings)

        # self.fm_second_order_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        # fm_second与第一层一样， 第一层用来计算 sum_k(sum_n(v_kn*x_k)**2) 第二层用来计算 sum_k(sum_n(v_kn**2*x_k**2))
        fm_second_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings)

        """
            init deep part
            就是两个带着BatchNorm和Dropout的全连接层
        """
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(dropout[i - 1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
        emb = self.fm_first_order_models[20]
        # print(Xi.size())
        for num in Xi[:, 20, :][0]:
            if num > self.feature_sizes[20]:
                print("index out")

        # fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(
        # self.fm_first_order_models)]
        # fm_first_order_emb_arr = [(emb(Xi[:, i, :]) * Xv[:, i])  for i, emb in enumerate(self.fm_first_order_models)]
        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_models):
            if i <= 12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        # fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
        # for i, emb in enumerate(self.fm_second_order_models)]
        # fm_second_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i])for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <= 12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)  # 横向拼接
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        # print("1",torch.sum(fm_first_order, 1).shape)
        # print("2",torch.sum(fm_second_order, 1).shape)
        # print("deep",torch.sum(deep_out, 1).shape)
        # print("bias",bias.shape)
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=5):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=self.dtype)

                total = model(xi, xv)
                #                print(total.shape)
                #                print(y.shape)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    self.check_accuracy(loader_val, model)
                    print()

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5).to(dtype=self.dtype)
                #                print(preds.dtype)
                #                print(y.dtype)
                #                print(preds.eq(y).cpu().sum())
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            #                print("successful")
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
