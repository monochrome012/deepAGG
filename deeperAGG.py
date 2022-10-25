import argparse
import math
import os
import random
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 设置随机种子和超参
def setup():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--batch_size", default=128, type=float)
    parser.add_argument("--num_workers", default=0, type=float)
    parser.add_argument("--p", default=18, type=int)
    parser.add_argument("--q", default=51, type=int)
    parser.add_argument("--k_p", default=3, type=int)
    parser.add_argument("--k_q", default=3, type=int)
    parser.add_argument("--nIter", default=2, type=int)
    parser.add_argument("--runs", default=10000, type=int)
    parser.add_argument("--train_percent", default=0.5, type=int)
    parser.add_argument(
        "--input_dir", default="./data/SpectralMethodsMeetEM/bluebird", type=str
    )
    parser.add_argument("--output_dir", default="./output128/deeperAGG/", type=str)

    args = parser.parse_args()

    args.output_dir = args.output_dir + os.path.basename(args.input_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args

# 给定目录，得到数据
def get_data(dir, train_percent=0.5):
    label_path = os.path.join(dir, "label.csv")
    truth_path = os.path.join(dir, "truth.csv")

    def read_csv(csv_path):
        res_list = []
        with open(csv_path) as f:
            lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]
            for i in range(1, len(lines)):
                res_list.append(
                    {lines[0][j]: int(lines[i][j]) for j in range(len(lines[i]))}
                )
        return res_list

    def get_data(workers, items, truth_dict):
        worker2id = {worker: id for id, worker in enumerate(workers)}
        item2id = {item: id for id, item in enumerate(items)}

        data = np.zeros((len(workers) + 1, len(items)), dtype="int64")
        data.fill(-1)
        for label in labels:
            if label["worker"] not in worker2id or label["item"] not in item2id:
                continue
            worker_id = worker2id[label["worker"]]
            item_id = item2id[label["item"]]
            data[worker_id][item_id] = label["label"]
        for item_id, item in enumerate(items):
            data[-1][item_id] = truth_dict[item]
        return data

    labels = read_csv(label_path)
    truths = read_csv(truth_path)

    workers = set()
    items = set()
    truth_dict = dict()
    responses = set()

    for label in labels:
        workers.add(label["worker"])
    for truth in truths:
        items.add(truth["item"])
        truth_dict[truth["item"]] = truth["truth"]
        responses.add(truth["truth"])

    workers = list(workers)
    items = list(items)
    random.shuffle(workers)
    random.shuffle(items)

    train_workers = workers[: int(len(workers) * train_percent)]
    test_workers = workers[int(len(workers) * train_percent) :]
    train_items = items[: int(len(items) * train_percent)]
    test_items = items[int(len(items) * train_percent) :]

    train_data = get_data(train_workers, train_items, truth_dict)
    test_data = get_data(test_workers, test_items, truth_dict)
    return train_data, test_data, len(responses)

# Dataset
class DeeperAGGDataset(Dataset):
    def __init__(self, data, a, p, q, runs=10000) -> None:
        self.data = data
        self.a = a
        self.p = p
        self.q = q
        self.runs = runs

    def __len__(self):
        return self.runs

    def __getitem__(self, index):
        row_index = np.arange(self.data.shape[0] - 1)
        col_index = np.arange(self.data.shape[1])
        np.random.seed(index)
        np.random.shuffle(row_index)
        np.random.shuffle(col_index)
        return (
            self.data[row_index[: self.p], :][:, col_index[: self.q]],
            self.data[-1, col_index[: self.q]],
        )

# 模型
class DeeperAGG(nn.Module):
    def __init__(self, p, q, a, k_p=3, k_q=3):
        super(DeeperAGG, self).__init__()
        self.p = p
        self.q = q
        self.k_p = k_p
        self.k_q = k_q
        self.a = a

        self.cor = nn.Sequential(
            nn.Linear(2 + k_p + k_q, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )
        self.ref = nn.Sequential(
            nn.Linear(self.p, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 1),
        )

    # 初始化G_hat，遵守少数服从多数的规则，得到初始的预测结果 
    def init_G_hat(self, M):
        self.G_hat = torch.stack(
            [
                torch.stack(
                    [
                        torch.argmax(torch.bincount(M_batch[:, j]))
                        for j in range(M_batch.shape[1])
                    ]
                )
                for M_batch in M
            ]
        )

    # 根据投票结果M和预测结果sefl.G_hat，进行特征提取
    def feature_represent(self, M):
        X = []
        for M_batch, g in zip(M, self.G_hat):
            M_check = torch.eq(M_batch, g)

            a_list = torch.sum(M_check, 1) / M_batch.shape[1]
            d_list = torch.sum(M_check, 0) / M_batch.shape[0]

            indexes_a = torch.split(torch.argsort(a_list), len(a_list) // self.k_p)
            indexes_d = torch.split(torch.argsort(d_list), len(d_list) // self.k_q)

            a_k_list = [
                torch.sum(M_check[:, indexes], 1) / len(indexes)
                for indexes in indexes_d
            ]
            d_k_list = [
                torch.sum(M_check[indexes, :], 0) / len(indexes)
                for indexes in indexes_a
            ]

            f_a = torch.tensor(
                [
                    [a_list[p], *[k_list[p] for k_list in a_k_list]]
                    for p in range(M_batch.shape[0])
                ]
            )
            f_d = torch.tensor(
                [
                    [d_list[q], *[k_list[q] for k_list in d_k_list]]
                    for q in range(M_batch.shape[1])
                ]
            )
            f_a = f_a.unsqueeze(1).repeat(1, M_batch.shape[1], 1)
            f_d = f_d.unsqueeze(0).repeat(M_batch.shape[0], 1, 1)
            X.append(torch.concat((f_a, f_d), 2))
        X = torch.stack(X)
        return X

    # 计算精度
    def accuacy(self, G_hat, G_true):
        G_check = torch.eq(G_hat, G_true)
        return torch.sum(G_check) / torch.numel(G_check)

    # 根据众包工作者的投票M和每个人的回答正确概率Y，得到最终特征矩阵U
    @torch.no_grad()
    def get_U(self, M, Y):
        U = torch.empty((M.shape[0], self.q, self.a, self.p))
        for i, (Y_batch, M_batch) in enumerate(zip(Y, M)):
            Y_batch = Y_batch.reshape(self.p, self.q)
            for j in range(self.q):
                for r in range(self.a):
                    pos_inds = torch.where(M_batch[:, j] == r)
                    neg_inds = torch.where(M_batch[:, j] != r)
                    U[i][j][r][pos_inds[0]] = Y_batch[pos_inds[0], j]
                    U[i][j][r][neg_inds[0]] = (1 - Y_batch[neg_inds[0], j]) / (
                        self.a - 1
                    )
        return U

    def forward(self, M, G_true=None, nIter=1):

        # 首先初始化G_hat
        self.init_G_hat(M)

        # loss需要累加
        loss_cor, loss_ref = torch.zeros(()), torch.zeros(())

        # 算法进行nIter次的迭代
        for _ in range(nIter):
            # 模块一：cor
            Y = self.feature_represent(M).to(torch.float32)
            Y = Y.reshape(Y.shape[0], -1, Y.shape[-1])
            Y = self.cor(Y).reshape(Y.shape[0], -1)

            # 模块二：ref
            U = self.get_U(M, Y)
            logits = F.softmax(
                self.ref(U.reshape(-1, U.shape[-1])).reshape(-1, self.a), 1
            )

            # 预测G_hat
            self.G_hat = torch.argmax(logits.reshape(U.shape[0], U.shape[1], -1), 2)

            # 计算损失
            if G_true is not None:
                G_check = torch.eq(M, G_true.unsqueeze(1)).to(torch.float32)
                G_check = G_check.reshape(G_true.shape[0], -1)
                loss_cor += F.binary_cross_entropy(Y, G_check) / nIter
                loss_ref += F.cross_entropy(logits, G_true.reshape(-1)) / nIter

        if G_true is not None:
            return loss_cor, loss_ref
        else:
            return self.G_hat


def C(n, m):

    return math.factorial(n) / (math.factorial(m) * math.factorial(n - m))


if __name__ == "__main__":
    # 设置超参
    args = setup()

    # 数据处理部分
    train_data, test_data, num_responses = get_data(args.input_dir, args.train_percent)

    if len(np.where(train_data == -1)[0]) > 0 or len(np.where(test_data == -1)[0]) > 0:
        print("不是完备的数据集！！！")
        exit()

    # print(train_data.shape, test_data.shape)
    # print(C(train_data.shape[0], args.p) * C(train_data.shape[1], args.q))
    # print(C(test_data.shape[0], args.p) * C(test_data.shape[1], args.q))
    pprint(args.__dict__)

    train_dataset = DeeperAGGDataset(
        train_data, num_responses, args.p, args.q, args.runs
    )
    test_dataset = DeeperAGGDataset(test_data, num_responses, args.p, args.q, args.runs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 模型训练
    model = DeeperAGG(args.p, args.q, num_responses, args.k_p, args.k_q)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loss, train_loss_cor, train_loss_ref, train_acc = [], [], [], []
    test_loss, test_loss_cor, test_loss_ref, test_acc = [], [], [], []
    best_acc = -1
    for epoch in range(args.epochs):
        # 训练
        model.train()
        loss_epoch, loss_cor_epoch, loss_ref_epoch, acc_epoch = [], [], [], []
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for M, G in tqdm_bar:
            loss_cor, loss_ref = model(M, G_true=G, nIter=args.nIter)
            loss = loss_cor + loss_ref

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = model.accuacy(model.G_hat, G)

            loss_epoch.append(loss.item())
            loss_cor_epoch.append(loss_cor.item())
            loss_ref_epoch.append(loss_ref.item())
            acc_epoch.append(acc.item())

            tqdm_bar.set_postfix(
                {
                    "train_loss": loss.item(),
                    "train_loss_cor": loss_cor.item(),
                    "train_loss_ref": loss_ref.item(),
                    "train_acc": acc.item(),
                }
            )

            # 保存每个epoch的曲线
            plt.title(f"output_{epoch+1}")
            plt.semilogy(loss_epoch, label="train_loss")
            plt.semilogy(loss_cor_epoch, label="train_loss_cor")
            plt.semilogy(loss_ref_epoch, label="train_loss_ref")
            plt.semilogy(acc_epoch, label="train_acc")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"output_{epoch+1}.png"), dpi=200)
            plt.close()

        train_loss.append(np.mean(loss_epoch))
        train_loss_cor.append(np.mean(loss_cor_epoch))
        train_loss_ref.append(np.mean(loss_ref_epoch))
        train_acc.append(np.mean(acc_epoch))

        # 测试
        model.eval()
        loss_epoch, loss_cor_epoch, loss_ref_epoch, acc_epoch = [], [], [], []
        tqdm_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        with torch.no_grad():
            for M, G in tqdm_bar:
                loss_cor, loss_ref = model(M, G_true=G, nIter=args.nIter)
                loss = loss_cor + loss_ref
                acc = model.accuacy(model.G_hat, G)

                loss_epoch.append(loss.item())
                loss_cor_epoch.append(loss_cor.item())
                loss_ref_epoch.append(loss_ref.item())
                acc_epoch.append(acc.item())

                tqdm_bar.set_postfix(
                    {
                        "test_loss": loss.item(),
                        "test_loss_cor": loss_cor.item(),
                        "test_loss_ref": loss_ref.item(),
                        "test_acc": acc.item(),
                    }
                )

        test_loss.append(np.mean(loss_epoch))
        test_loss_cor.append(np.mean(loss_cor_epoch))
        test_loss_ref.append(np.mean(loss_ref_epoch))
        test_acc.append(np.mean(acc_epoch))

        # 保存变化曲线
        print("*" * 42)
        print(f"Epoch {epoch+1}:")
        print(f"train_loss = {train_loss[-1]}")
        print(f"train_loss_cor = {train_loss_cor[-1]}")
        print(f"train_loss_ref = {train_loss_ref[-1]}")
        print(f"train_acc = {train_acc[-1]}")
        print(f"test_loss = {test_loss[-1]}")
        print(f"test_loss_cor = {test_loss_cor[-1]}")
        print(f"test_loss_ref = {test_loss_ref[-1]}")
        print(f"test_acc = {test_acc[-1]}")
        print("*" * 42)

        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"best.pth.tar"),
            )

        plt.title(f"output")
        plt.semilogy(train_loss, label="train_loss")
        plt.semilogy(train_loss_cor, label="train_loss_cor")
        plt.semilogy(train_loss_ref, label="train_loss_ref")
        plt.semilogy(test_loss, label="test_loss")
        plt.semilogy(test_loss_cor, label="test_loss_cor")
        plt.semilogy(test_loss_ref, label="test_loss_ref")
        plt.semilogy(train_acc, label="train_acc")
        plt.semilogy(test_acc, label="test_acc")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f"output.png"), dpi=200)
        plt.close()
