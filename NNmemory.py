import torch
import functools
import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBankModule(torch.nn.Module):
    def __init__(self, size: int = 2**16, cluster_num: int = 0):
        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f"Illegal memory bank size {size}, must be non-negative."
            raise ValueError(msg)

        self.size = size
        self.cluster_num = cluster_num  #聚类个数
        self.register_buffer(
            "bank", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "label", tensor=torch.empty(0, dtype=torch.int64), persistent=False
        )

    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        self.bank = torch.randn(dim, self.size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)
        #self.label = torch.randn(1, self.size).type_as(self.bank)
        self.label = torch.randint(self.cluster_num,(1,self.size)).type_as(self.label)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor, batch_label: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            num = ptr + batch_size - self.size
            self.bank[:, ptr:] = batch[: self.size - ptr].T.detach()
            self.label[:, ptr:] = batch_label[: self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
            if num != 0:
                self.bank[:, self.bank_ptr[0]:self.bank_ptr[0] + num] = batch[(self.size - ptr) : ].T.detach()
                self.label[:, self.bank_ptr[0]:self.bank_ptr[0] + num] = batch_label[(self.size - ptr):].T.detach()
                self.bank_ptr[0] = self.bank_ptr[0] + num
        else:
            self.bank[:, ptr : ptr + batch_size] = batch.T.detach()
            self.label[:, ptr : ptr + batch_size] = batch_label.T.detach()
            self.bank_ptr[0] = ptr + batch_size


    def forward(
        self, output: torch.Tensor, labels: torch.Tensor = None , update: bool = False
    ):
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)


        # query and update memory bank
        bank = self.bank.clone().detach()
        label = self.label.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output,labels)

        return output, bank


    # def cluster_centre(self, label):
    #     bank = self.bank.cpu().clone().detach()
    #     dim = bank.size(0)
    #     labels = self.label.cpu().clone().detach()
    #
    #     mask = labels.eq(label)
    #     if torch.sum(mask) != 0:
    #         bank1 = torch.masked_select(bank, mask).view(dim,-1)
    #     else:
    #         return 0
    #     # sample = [np.array(i) for i in bank if i[-1]==label]
    #     # sample = torch.tensor(sample, dtype=torch.float32).to(device)
    #     centre = torch.mean(bank1, dim=1)
    #     return centre


class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    """

    def __init__(self, size: int = 2**16, cluster_num: int = 0):
        super(NNMemoryBankModule, self).__init__(size,cluster_num)

    def forward(self, output_: torch.Tensor, labels: torch.Tensor = None, update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """

        output, bank = super(NNMemoryBankModule, self).forward(output_, labels, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(
            bank, dim=0, index=index_nearest_neighbours
        )

        return nearest_neighbours

    # def cluster_centre(self, label):
    #     centre = super(NNMemoryBankModule, self).cluster_centre(label)
    #     return centre

    def getcentre(self, cluster_num):
        centre_all = None
        bank = self.bank.cpu().clone().detach()
        dim = bank.size(0)
        labels = self.label.cpu().clone().detach()

        for i in range(cluster_num):
            mask = labels.eq(i)
            bank1 = torch.masked_select(bank, mask).view(dim, -1)
            if bank1.size(1)>0:
                centre = torch.mean(bank1, dim=1).unsqueeze(0)
                if centre_all == None:
                    centre_all = centre
                else:
                    centre_all = torch.cat((centre_all, centre), dim=0)
        return centre_all.T
