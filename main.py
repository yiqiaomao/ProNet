import argparse
import torch
import numpy as np
import utils1
from dataset import Dateset_mat, data_loder
from tqdm import trange
from model import En_decoder, Encoder_shared, UD_constraint
import copy
import warnings
from NNmemory import NNMemoryBankModule
import itertools
from mutual_information import cluster_centre_distillation
from lightly.loss import NTXentLoss
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
data_name = 'name'

parser.add_argument("--dataset_root", default=r'./caltech-2v/', type=str)
parser.add_argument("--modality_num", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr_1", type=float, default=0.001)
parser.add_argument("--all_epochs", type=int, default=20)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--pretrain_epochs", type=int, default=200)

parser.add_argument("--feature_dimension", type=int, default=128)
parser.add_argument("--project_dimension", type=int, default=128)
config = parser.parse_args()
config.max_ACC = 0

# NTX_loss = NTXentLoss()
CE_loss = torch.nn.CrossEntropyLoss().to(device)
MSE_loss = torch.nn.MSELoss().to(device)
NTX_loss = NTXentLoss()
loss_NTXent = NTXentLoss(config.batch_size)

def run():
    Dataset = Dateset_mat(config.dataset_root, config.modality_num)
    dataset = Dataset.getdata()
    label = np.array(dataset[dataset.__len__() - 1])
    label = np.squeeze(label)
    cluster_num = max(label) + 1
    print("clustering number: ", cluster_num)
    modality_size = [dataset[i].shape[1] for i in range(config.modality_num)]
    print(len(dataset[0]))

    modality_data = [torch.tensor(dataset[i], dtype=torch.float32).to(device) for i in range(config.modality_num)]

    data = data_loder(config.batch_size, config.modality_num)
    data.get_data(dataset)

    memory_bank = NNMemoryBankModule(size=int(8192), cluster_num=cluster_num)
    memory_bank.to(device)

    S_encoder = Encoder_shared(config.project_dimension, cluster_num).to(device)
    all_encoder = [En_decoder(modality_size[i], config.feature_dimension, config.project_dimension).to(device) for i in
                   range(config.modality_num)]

    pretrain(all_encoder, data)
    for all_epoch in range(config.all_epochs):
        print(' ')
        for modal_num in range(config.modality_num):
            frozen_S_encoder = copy.deepcopy(S_encoder)
            frozen_S_encoder.eval()

            if modal_num + 1 < config.modality_num:
                a, b = modal_num, modal_num + 1
            else:
                a, b = modal_num, 0
            e_dcoder1, e_dcoder2 = all_encoder[a], all_encoder[b]
            parame = itertools.chain(e_dcoder1.parameters(), e_dcoder2.parameters(), S_encoder.parameters())
            optimiser = torch.optim.Adam(parame, lr=config.lr)
            for epoch in range(config.num_epochs):
                e_dcoder1.train()
                e_dcoder2.train()
                S_encoder.train()
                for data_ in data:
                    data_1, data_2 = data_[a].to(device), data_[b].to(device)
                    _, z1, _ = e_dcoder1(data_1)
                    _, z2, _ = e_dcoder2(data_2)
                    f1, f2, p1, p2, clustering1, clustering2 = S_encoder(z1, z2)

                    pseudo_label1 = torch.argmax(clustering1, dim=1).unsqueeze(-1)
                    pseudo_label2 = torch.argmax(clustering2, dim=1).unsqueeze(-1)

                    f1_nn_labeled = memory_bank(f1, pseudo_label1, update=True)
                    f2_nn_labeled = memory_bank(f2, pseudo_label2, update=False)

                    loss_NTX_F = NTX_loss(f1_nn_labeled, p2) + NTX_loss(f2_nn_labeled, p1)
                    loss_NTX_C = NTX_loss(clustering1, clustering2)

                    UDC = UD_constraint(clustering1)
                    UDC = UDC.to(device)
                    loss_UDC = CE_loss(clustering1, UDC)


                    with torch.no_grad():
                        before_f1, before_f2, _, _, _, _ = frozen_S_encoder(z1, z2)
                        cluster_centre = memory_bank.getcentre(cluster_num).to(device)
                        loss_centre = cluster_centre_distillation(before_f1, f1, cluster_centre.T)

                    loss = loss_NTX_F + loss_NTX_C + loss_UDC + loss_centre
                    loss.backward()
                    optimiser.step()

                if epoch % 10 == 0:
                    acc, nmi, ari = get_S_ACC(e_dcoder1, e_dcoder2, S_encoder, modality_data[a], modality_data[b], label, epoch)
                    print(f"All_epoch:{all_epoch:} ,Modal_num:{modal_num+1:} ,Train_epoch: {epoch:>02}, ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")


def pretrain(all_encoder, data):
    for epoch in trange(config.pretrain_epochs):
        total_loss = 0
        for modal_num in range(config.modality_num):
            encoder = all_encoder[modal_num]
            optimiser = torch.optim.Adam(encoder.parameters(), lr=config.lr_1)
            encoder.train()
            encoder.zero_grad()
            for data_ in data:
                data_ = data_[modal_num].to(device)
                _, _, x_dec = encoder(data_)
                loss_mse = MSE_loss(data_, x_dec)
                total_loss += loss_mse
                loss_mse.backward()
                optimiser.step()



def get_S_ACC(e_dcoder1, e_dcoder2, S_encoder, data1, data2, label, epoch):
    e_dcoder1.eval()
    e_dcoder2.eval()
    S_encoder.eval()
    _, p1, _ = e_dcoder1(data1)
    _, p2, _ = e_dcoder2(data2)
    _, _, _, _, clustering, _ = S_encoder(p1, p2)
    pre_label = np.array(clustering.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc = utils1.metrics.acc(pre_label, label)
    nmi = utils1.metrics.nmi(pre_label, label)
    ari = utils1.metrics.ari(pre_label, label)
    return acc, nmi, ari





if __name__ == '__main__':
    run()
