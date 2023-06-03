import numpy as np
from typing import List, Union, Tuple

import torch
from torch import nn

class GraphFilter(nn.Module):
    def __init__(self, k: int, f_in=1, f_out=1, f_edge=1):
        """
        A graph filter layer.
        Args:
            k: The number of filter taps (the max diffusive order of the convolution).
            f_in: The number of input features.
            f_out: The number of output features.
            f_edge: The number of edge features.
        """
        super().__init__()
        self.k = k
        self.f_in = f_in
        self.f_out = f_out
        self.f_edge = f_edge

        self.weight = nn.Parameter(torch.ones(self.f_out, self.f_edge, self.k, self.f_in))
        self.bias = nn.Parameter(torch.zeros(self.f_out, 1))

        # Initialize learning parameters
        stdv = 1. / np.sqrt(self.k)
        torch.nn.init.uniform_(self.weight, -stdv, stdv)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor):
        """
        A pass forward through the graph filter.
        Args:
            x: The input signal of shape B x G x N, where B is the batch size, G is the number of input node features,
               and N is the number of nodes.
            S: The graph shift operator of shape B x E x N x N, where E is the number of edge features.
        """

        B = x.shape[0]
        E = self.f_edge
        F = self.f_out
        G = self.f_in
        N = S.shape[-1]
        K = self.k   # number of filter taps

        h = self.weight
        b = self.bias

        # Now, we have x in B x G x N and S in B x E x N x N, and we want to come up
        # with matrix multiplication that yields z = x * S with shape
        # B x E x K x G x N.
        # For this, we first add the corresponding dimensions
        x = x.reshape([B, 1, G, N])
        S = S.reshape([B, E, N, N])
        z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1)  # This is for k = 0
        # We need to repeat along the E dimension, because for k=0, S_{e} = I for
        # all e, and therefore, the same signal values have to be used along all
        # edge feature dimensions.
        for k in range(1, K):
            x = torch.matmul(x, S)  # B x E x G x N
            xS = x.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
            z = torch.cat((z, xS), dim=2)  # B x E x k x G x N
        # This output z is of size B x E x K x G x N
        # Now we have the x*S_{e}^{k} product, and we need to multiply with the
        # filter taps.
        # We multiply z on the left, and h on the right, the output is to be
        # B x N x F (the multiplication is not along the N dimension), so we reshape
        # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
        # always reshape the last dimensions), and then make h be E x K x G x F and
        # reshape it to EKG x F, and then multiply
        y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E * K * G]),
                         h.reshape([F, E * K * G]).permute(1, 0)).permute(0, 2, 1)
        # And permute again to bring it from B x N x F to B x F x N.
        # Finally, add the bias
        if b is not None:
            y = y + b
        return y

class MultiGraphFilter(nn.Module):
    def __init__(self, GSOs, f_in=1, f_out=1, f_edge=1):
        """
        A multigraph filter layer.
        Args:
            GSOs: A list of E graph shift operators, each of size N x N.
            f_in: The number of input features.
            f_out: The number of output features.
            f_edge: The number of edge features.
        """
        super().__init__()
        self.GSOs = GSOs
        self.f_in = f_in
        self.f_out = f_out
        self.f_edge = f_edge
        self.weight = nn.Parameter(torch.ones(self.f_out, self.f_edge, len(self.GSOs) + 1, self.f_in))
        self.bias = nn.Parameter(torch.zeros(self.f_out, 1))

        # Initialize learning parameters
        torch.nn.init.normal_(self.weight, 0.05, 0.01)
        torch.nn.init.zeros_(self.bias)

    def addGSO(self, GSOs):
        self.GSOs = GSOs

    def forward(self, x: torch.Tensor):
        """
        A pass forward through the multigraph filter.
        Args:
            x: The input signal of shape B x G x N, where B is the batch size, G is the number of input node features,
               and N is the number of nodes.
        """
        B = x.shape[0]
        E = self.f_edge
        F = self.f_out
        G = self.f_in
        N = self.GSOs[0].shape[-1]

        h = self.weight
        b = self.bias

        S = []
        for GSO in self.GSOs:
            S.append(GSO.repeat(B, E, 1, 1).reshape([B, E, N, N]))

        K = len(S) + 1

        # Reshape x to perform the multigraph convolution
        # Create z to store convolved signal in
        x = x.reshape([B, 1, G, N])
        z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1)

        for k in range(len(S)):
            xS = torch.matmul(x, S[k])  # B x E x G x N
            xS = xS.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
            z = torch.cat((z, xS), dim=2)  # B x E x k x G x N
            del xS
        del S
        # This output z is of size B x E x K x G x N
        # Now we have multigraph polynomial and we need to multiply with the filter taps.
        # We multiply z on the left, and h on the right, the output is to be
        # B x N x F (the multiplication is not along the N dimension), so we reshape
        # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
        # always reshape the last dimensions), and then make h be E x K x G x F and
        # reshape it to EKG x F, and then multiply
        y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E * K * G]),
                         h.reshape([E * K * G, F])).permute(0,2,1)
        del z
        # And permute again to bring it from B x N x F to B x F x N.
        # Finally, add the bias
        if b is not None:
            y = y + b
        return y

class GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 ks: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1),
                 f_edge: int = 1,
                 ):
        """
        An L-layer graph neural network. Uses Sigmoid activation for each layer.

        Args:
            ks: [K_1,...,K_L]. On ith layer, K_{i} is the number of filter taps.
            fs: [F_1,...,F_L]. On ith layer, F_{i} and F_{i+1} are the number of input and output features,
             respectively.
            f_edge: The number of edge features.
        """
        super().__init__()
        self.n_layers = len(ks)

        self.layers = []
        for i in range(self.n_layers):
            f_in = fs[i]
            f_out = fs[i + 1]
            k = ks[i]
            gfl = GraphFilter(k, f_in, f_out, f_edge)
            activation = torch.nn.Sigmoid()
            self.layers += [gfl, activation]
            self.add_module(f"gfl{i}", gfl)
            self.add_module(f"activation{i}", activation)


        # Readout layer
        lin = nn.Linear(fs[len(fs) - 1], 2)
        act = torch.nn.Sigmoid()
        self.layers += [lin, act]
        self.add_module(f"lin", lin)
        self.add_module(f"finAct", act)

    def forward(self, x, S):
        """
        A pass forward through the graph neural network.
        Args:
            x: The input signal of shape B x G x N, where B is the batch size, G is the number of input node features,
               and N is the number of nodes.
            S: The graph shift operator of shape B x E x N x N, where E is the number of edge features.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, S) if i % 2 == 0 else layer(x)
        return torch.abs(x)

class MultiGraphNeuralNetwork(nn.Module):
    def __init__(self,
                 GSOs,
                 maxOrders: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1),
                 ):
        """
        An L-layer multigraph neural network. Uses Sigmoid activation for each layer.

        Args:
            GSOs: A list of E graph shift operators, each of size N x N.
            maxOrders: [K_1,...,K_L]. On ith layer, K_{i} is the max diffusive order of the multigraph polynomial.
            fs: [F_1,...,F_L]. On ith layer, F_{i} and F_{i+1} are the number of input and output features,
             respectively.
        """
        super().__init__()
        self.n_layers = len(maxOrders)
        self.maxOrders = maxOrders

        self.S = torch.tensor(GSOs)  # E x N x N
        self.E = GSOs.shape[0]

        self.term_idxs = [self.generate_term_idxs(self.E, d) for d in maxOrders]
        self.GSOs = [self.find_term_matrices(GSOs, self.term_idxs[layer]) for layer in range(len(self.term_idxs))]
        self.layers = []
        for i in range(self.n_layers):
            f_in = fs[i]
            f_out = fs[i + 1]
            gfl = MultiGraphFilter(self.GSOs[i], f_in, f_out)
            activation = torch.nn.Sigmoid()
            self.layers += [gfl, activation]
            self.add_module(f"mgfl{i}", gfl)
            self.add_module(f"activation{i}", activation)

        # Readout layer
        lin = nn.Linear(fs[len(fs) - 1], 2)
        act = torch.nn.Sigmoid()
        self.layers += [lin, act]
        self.add_module(f"lin", lin)
        self.add_module(f"finAct", act)

    def generate_term_idxs(self, numGSOs, depth):
        frontier_terms = [tuple([i]) for i in range(numGSOs)]
        computed_terms = set()
        while len(frontier_terms) > 0:
          lead_term = frontier_terms.pop()
          computed_terms.add(lead_term)

          # Multiply by the left of with each GSO
          for gso in range(numGSOs):
            if len(lead_term) < depth:
              frontier_terms.append(tuple([gso] + list(lead_term)))
        return computed_terms

    def find_term_matrices(self, GSOs, term_idxs):
        n = GSOs[0].shape[1]
        GSO_interactions = []
        for i,term in enumerate(term_idxs):
          mat_term = GSOs[term[0],:,:]
          if len(term) > 1:
            for j in range(1,len(term)):
              mat_term = mat_term @ GSOs[term[j],:,:]
          GSO_interactions.append(torch.tensor(mat_term))
        return GSO_interactions

    def changeGSO(self, GSOs):
        """
        We use this to change the GSO, using the same graph filters.

        Args:
            GSOs: A list of E graph shift operators, each of size N x N.

        """
        S = torch.tensor(GSOs)  # E x N x N
        # Check that the new GSO has the correct shape
        assert len(S.shape) == 3
        assert S.shape[1] == S.shape[2]  # E x N x N

        self.S = S
        self.term_idxs = [self.generate_term_idxs(self.E, d) for d in self.maxOrders]
        self.GSOs = [self.find_term_matrices(GSOs, self.term_idxs[layer]) for layer in range(len(self.term_idxs))]
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                layer.addGSO(self.GSOs[i // 2])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return torch.abs(x)
