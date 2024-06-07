import scipy.io
import torch
import numpy as np
import random
import torch.utils.data as data
import copy


class Dateset_mat():
    def __init__(self, data_path, view_num):
        self.view_num = view_num
        try:
            self.label = scipy.io.loadmat(data_path + r"/labels.mat")
        except:
            self.label = scipy.io.loadmat(data_path + r"/L.mat")

        if view_num == 2:
            try:
                self.v1 = scipy.io.loadmat(data_path + r"/view1.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/view2.mat")
            except:
                self.v1 = scipy.io.loadmat(data_path + r"/img.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/txt.mat")
        elif view_num == 3:
            try:
                self.v1 = scipy.io.loadmat(data_path + r"/audio.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/text.mat")
                try:
                    self.v3 = scipy.io.loadmat(data_path + r"/vision.mat")
                except:
                    self.v3 = scipy.io.loadmat(data_path + r"/video.mat")
            except:
                try:
                    self.v1 = scipy.io.loadmat(data_path + r"/view1.mat")
                    self.v2 = scipy.io.loadmat(data_path + r"/view2.mat")
                    self.v3 = scipy.io.loadmat(data_path + r"/view4.mat")
                except:
                    self.v1 = scipy.io.loadmat(data_path + r"/view1.mat")
                    self.v2 = scipy.io.loadmat(data_path + r"/view2.mat")
                    self.v3 = scipy.io.loadmat(data_path + r"/view3.mat")
        elif view_num == 4:
            try:
                self.v1 = scipy.io.loadmat(data_path + r"/view1.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/view2.mat")
                self.v3 = scipy.io.loadmat(data_path + r"/view3.mat")
                self.v4 = scipy.io.loadmat(data_path + r"/view4.mat")
            except:
                self.v1 = scipy.io.loadmat(data_path + r"/image-vgg.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/image-resnet.mat")
                self.v3 = scipy.io.loadmat(data_path + r"/text-bert.mat")
                self.v4 = scipy.io.loadmat(data_path + r"/text-xlnet.mat")
        elif view_num == 5:
            try:
                self.v1 = scipy.io.loadmat(data_path + r"/view1.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/view2.mat")
                self.v3 = scipy.io.loadmat(data_path + r"/view3.mat")
                self.v4 = scipy.io.loadmat(data_path + r"/view4.mat")
                self.v5 = scipy.io.loadmat(data_path + r"/view5.mat")
            except:
                self.v1 = scipy.io.loadmat(data_path + r"/v1.mat")
                self.v2 = scipy.io.loadmat(data_path + r"/v2.mat")
                self.v3 = scipy.io.loadmat(data_path + r"/v3.mat")
                self.v4 = scipy.io.loadmat(data_path + r"/v4.mat")

    def getdata(self):
        self.data = []
        if self.view_num == 2:
            try:
                self.data.append(self.v1["img"])
                self.data.append(self.v2["txt"])
            except:
                self.data.append(self.v1["fea"].astype(float))
                self.data.append(self.v2["fea"].astype(float))
        elif self.view_num == 3:
            try:
                self.data.append(self.v1["fea"].astype(float))
                self.data.append(self.v2["fea"].astype(float))
                self.data.append(self.v3["fea"].astype(float))
            except:
                self.data.append(self.v1["view1"])
                self.data.append(self.v2["view2"])
                self.data.append(self.v3["view3"])
        elif self.view_num == 4:
            try:
                self.data.append(self.v1["fea"].astype(float))
                self.data.append(self.v2["fea"].astype(float))
                self.data.append(self.v3["fea"].astype(float))
                self.data.append(self.v4["fea"].astype(float))
            except:
                self.data.append(self.v1["image"].astype(float))
                self.data.append(self.v2["image"].astype(float))
                self.data.append(self.v3["text"].astype(float))
                self.data.append(self.v4["text"].astype(float))
        elif self.view_num == 5:
            self.data.append(self.v1["fea"].astype(float))
            self.data.append(self.v2["fea"].astype(float))
            self.data.append(self.v3["fea"].astype(float))
            self.data.append(self.v4["fea"].astype(float))
            self.data.append(self.v5["fea"].astype(float))
        try:
            if min(self.label["L"]) == 0:
                self.label["L"] = self.label["L"] + 1
            self.data.append(self.label["L"])
        except:
            self.data.append(self.label["labels"])
        return self.data


class data_loder(data.Dataset):
    def __init__(self, batch_size, view_num):
        self.batch_size = batch_size
        self.view_num = view_num
        self.getdata = {
            2: self.get_data_v2,
            3: self.get_data_v3,
            4: self.get_data_v4,
            5: self.get_data_v5
        }

    def get_data(self, input):
        self.fix_seed(1120)
        tempgetdata = self.getdata.get(self.view_num)
        tempgetdata(input)

    def __getitem__(self, index):
        if self.view_num == 4:
            v1, v2, v3, v4, label = np.array(self.data1[index]), np.array(self.data2[index]), self.data3[index], self.data4[index], self.data5[index]
            v1 = torch.tensor(v1, dtype=torch.float32)
            v2 = torch.tensor(v2, dtype=torch.float32)
            v3 = torch.tensor(v3, dtype=torch.float32)
            v4 = torch.tensor(v4, dtype=torch.float32)
            label = np.array(label) - 1
            label = np.squeeze(label)
            return v1, v2, v3, v4, label
        elif self.view_num == 5:
            v1, v2, v3, v4, v5, label = np.array(self.data1[index]), np.array(self.data2[index]), self.data3[index], self.data4[index], self.data5[index], self.data6[index]
            v1 = torch.tensor(v1, dtype=torch.float32)
            v2 = torch.tensor(v2, dtype=torch.float32)
            v3 = torch.tensor(v3, dtype=torch.float32)
            v4 = torch.tensor(v4, dtype=torch.float32)
            v5 = torch.tensor(v5, dtype=torch.float32)
            label = np.array(label) - 1
            label = np.squeeze(label)
            return v1, v2, v3, v4, v5, label
        elif self.view_num == 2:
            v1, v2, label = np.array(self.data1[index]), np.array(self.data2[index]), self.data5[index]
            v1 = torch.tensor(v1, dtype=torch.float32)
            v2 = torch.tensor(v2, dtype=torch.float32)
            label = np.array(label)-1
            label = np.squeeze(label)
            return v1, v2, label
        elif self.view_num == 3:
            v1, v2, v3, label = np.array(self.data1[index]), np.array(self.data2[index]), self.data3[index], self.data5[index]
            v1 = torch.tensor(v1, dtype=torch.float32)
            v2 = torch.tensor(v2, dtype=torch.float32)
            v3 = torch.tensor(v3, dtype=torch.float32)
            label = np.array(label) - 1
            label = np.squeeze(label)
            return v1, v2, v3, label

    def get_data_v2(self, input):
        v1, v2, label = input[0], input[1], input[2]
        size, size1 = v1.__len__(), v2.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        v1 = v1[shuffle_ix]
        v2 = v2[shuffle_ix]
        label = label[shuffle_ix]

        # img, txt, label = random.shuffle(img, txt, label)
        assert (size == size1)
        data1, data2, data5 = [], [], []
        alldata1, alldata2, alldata5 = [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(v1[i])
                data2.append(v2[i])
                data5.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d5 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data5)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata5.append(d5)

                data1.clear()
                data2.clear()
                data5.clear()
        self.data1 = alldata1
        self.data2 = alldata2
        self.data5 = alldata5

    def get_data_v4(self, input):
        v1, v2, v3, v4, label = input[0], input[1], input[2], input[3], input[4]
        size, size1 = v1.__len__(), v2.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        v1 = v1[shuffle_ix]
        v2 = v2[shuffle_ix]
        v3 = v3[shuffle_ix]
        v4 = v4[shuffle_ix]
        label = label[shuffle_ix]

        # img, txt, label = random.shuffle(img, txt, label)
        assert (size == size1)
        data1, data2, data3, data4, data5 = [], [], [], [], []
        alldata1, alldata2, alldata3, alldata4, alldata5 = [], [], [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(v1[i])
                data2.append(v2[i])
                data3.append(v3[i])
                data4.append(v4[i])
                data5.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d3, d4, d5 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data3), copy.deepcopy(data4), copy.deepcopy(data5)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata3.append(d3)
                alldata4.append(d4)
                alldata5.append(d5)

                data1.clear()
                data2.clear()
                data3.clear()
                data4.clear()
                data5.clear()
        self.data1 = alldata1
        self.data2 = alldata2
        self.data3 = alldata3
        self.data4 = alldata4
        self.data5 = alldata5

    import random
    def fix_seed(self, seed = 0):
        print('seed:', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def get_data_v3(self, input):
        v1, v2, v3, label = input[0], input[1], input[2], input[3]
        size, size1 = v1.__len__(), v2.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        v1 = v1[shuffle_ix]
        v2 = v2[shuffle_ix]
        v3 = v3[shuffle_ix]
        label = label[shuffle_ix]

        # img, txt, label = random.shuffle(img, txt, label)
        assert (size == size1)
        data1, data2, data3, data5 = [], [], [], []
        alldata1, alldata2, alldata3, alldata5 = [], [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(v1[i])
                data2.append(v2[i])
                data3.append(v3[i])
                data5.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d3, d5 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data3), copy.deepcopy(data5)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata3.append(d3)
                alldata5.append(d5)

                data1.clear()
                data2.clear()
                data3.clear()
                data5.clear()
        self.data1 = alldata1
        self.data2 = alldata2
        self.data3 = alldata3
        self.data5 = alldata5


    def get_data_v5(self, input):
        v1, v2, v3, v4, v5, label = input[0], input[1], input[2], input[3], input[4], input[5]
        size, size1 = v1.__len__(), v2.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        v1 = v1[shuffle_ix]
        v2 = v2[shuffle_ix]
        v3 = v3[shuffle_ix]
        v4 = v4[shuffle_ix]
        v5 = v5[shuffle_ix]
        label = label[shuffle_ix]

        # img, txt, label = random.shuffle(img, txt, label)
        assert (size == size1)
        data1, data2, data3, data4, data5, data6 = [], [], [], [], [], []
        alldata1, alldata2, alldata3, alldata4, alldata5, alldata6 = [], [], [], [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(v1[i])
                data2.append(v2[i])
                data3.append(v3[i])
                data4.append(v4[i])
                data5.append(v5[i])
                data6.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d3, d4, d5, d6 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data3), copy.deepcopy(data4), copy.deepcopy(data5), copy.deepcopy(data6)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata3.append(d3)
                alldata4.append(d4)
                alldata5.append(d5)
                alldata6.append(d6)

                data1.clear()
                data2.clear()
                data3.clear()
                data4.clear()
                data5.clear()
                data6.clear()

        self.data1 = alldata1
        self.data2 = alldata2
        self.data3 = alldata3
        self.data4 = alldata4
        self.data5 = alldata5
        self.data6 = alldata6
