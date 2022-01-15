import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
from typing import List, Union, Tuple, Callable
import numpy as np

class MultiGraphFilter(nn.Module):
    def __init__(self, GSOs, f_in=1, f_out=1, f_edge=1, bias = True):
        """
        A multigraph filter layer.
        Args:
            numGSOs: Number of Graph Shift Operators.
            depth: Max order on interaction terms.
            f_in: The number of input features.
            f_out: The number of output features.
        """
        assert f_edge == 1
        super().__init__()
        self.GSOs = GSOs
        self.f_in = f_in
        self.f_out = f_out
        self.f_edge = f_edge

        self.weight = nn.Parameter(torch.ones(self.f_out, self.f_edge, len(self.GSOs) + 1, self.f_in))
        torch.nn.init.normal_(self.weight, 0, 3)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.f_out, 1)) 
            torch.nn.init.normal_(self.bias, 0, 3)

    def to(self, *args, **kwargs):
        # Only the filter taps and the weights are registered as
        # parameters, so we need to move gsos ourselves.
        self.weight = self.gso.to(*args, **kwargs)
        self.bias = self.gso.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor, GSOs = None):
        # GSOs is never used, just used to match style of other architecures
        S = self.GSOs
        batch_size = x.shape[0]

        B = batch_size
        E = self.f_edge
        F = self.f_out
        G = self.f_in
        N = S[0].shape[-1]  # number of nodes
        K = len(S) + 1

        h = self.weight
        b = self.bias

        # Now, we have x in B x G x N and S in E x N x N, and we want to come up
        # with matrix multiplication that yields z = x * S with shape
        # B x E x K x G x N.
        # For this, we first add the corresponding dimensions
        x = torch.unsqueeze(x,1).permute(0,1,3,2)
        z = torch.unsqueeze(x,2).repeat(1, E, 1, 1, 1) 
        # We need to repeat along the E dimension, because for k=0, S_{e} = I for
        # all e, and therefore, the same signal values have to be used along all
        # edge feature dimensions.
        for k in range(len(S)):
            xNew = torch.zeros(B,E,G,N).to(device)
            for batch in range(B):
              for e in range(E):
                # e = 0 for x since it is same signal
                xNew[batch,e,:,:] = (S[k] @ x[batch,0,:,:].T).T
            x = xNew
            xS = x.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
            del xNew
            z = torch.cat((z, xS), dim=2)  # B x E x k x G x N}
            del xS
          
        # This output z is of size B x E x K x G x N
        # Now we have the x*S_{e}^{k} product, and we need to multiply with the
        # filter taps.
        # We multiply z on the left, and h on the right, the output is to be
        # B x N x F (the multiplication is not along the N dimension), so we reshape
        # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
        # always reshape the last dimensions), and then make h be E x K x G x F and
        # reshape it to EKG x F, and then multiply
        y = torch.matmul(z.reshape(B, N, E, K, G).reshape([B, N, E * K * G]), h.reshape([E * K * G, F])).reshape(B,F,N)
        # y = z.squeeze().reshape(B,G,N)
        # And permute again to bring it from B x N x F to B x F x N.
        # Finally, add the bias
        if b is not None:
            y = y + b
        return y.reshape(B,N,F)