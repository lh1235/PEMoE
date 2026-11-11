import torch
from torch_geometric.data import Data, Dataset

class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default', node_ids=None):
        if mode == 'default':
            self.x = inputs[split+'_x']
            self.y = inputs[split+'_y'] # [T, Len, N]
        else:
            self.x = x
            self.y = y
        self.node_ids = (torch.arange(self.x.shape[-1]) if node_ids is None
                         else torch.as_tensor(node_ids, dtype=torch.long).cpu())


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        data = Data(x=x, y=y, node_ids=self.node_ids)
        return data

    def len(self):
        return self.__len__()

    def get(self, index):
        return self.__getitem__(index)
