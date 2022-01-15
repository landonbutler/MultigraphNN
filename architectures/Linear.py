import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
from typing import List, Union, Tuple, Callable
import numpy as np
from scipy.sparse import coo_matrix
from .GraphFilter import GraphFilter

class Linear(nn.Module):
    def __init__(self,
                 GSOs,
                 ks: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1),
                 f_edge = 1,
                 idxTrainMovie = 0,
                 device = 'cpu'):
        super().__init__()
        deviceGSOs = []
        N = GSOs[0].shape[0]
        
        for ind, GSO in enumerate(GSOs):
          coo = coo_matrix(GSO)
          values = coo.data
          indices = np.vstack((coo.row, coo.col))

          i = torch.LongTensor(indices)
          v = torch.DoubleTensor(values)
          shape = coo.shape

          GSOsparse = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
          deviceGSOs.append(GSOsparse.to(device))
        
        self.F = fs
        self.K = ks
        self.f_edge = f_edge
        self.penaltyMultiplier = 0

        self.S = deviceGSOs
        self.idx = idxTrainMovie
        self.n_layers = len(ks)
        self.convLayers = []
        for i in range(len(ks)):
            f_in = fs[i]
            f_out = fs[i + 1]
            k = ks[i]
            gfl = GraphFilter(k, f_in, f_out, self.f_edge, device = device)
            self.convLayers += [gfl]
            self.add_module(f"gfl{i}", gfl)

        
    def forward(self, x):
        for i, layer in enumerate(self.convLayers):
            x = layer(x, self.S)
        return x[:,self.idx,:]