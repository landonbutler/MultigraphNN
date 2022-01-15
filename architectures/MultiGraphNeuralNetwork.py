import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
from typing import List, Union, Tuple, Callable
import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix
from .MultiGraphFilter import MultiGraphFilter

class MultiGraphNeuralNetwork(nn.Module):
    def __init__(self, 
                 GSOs,
                 depths: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1),
                 readout: Union[List[int], Tuple[int]] = (1, 1),
                 nonlinearity = torch.nn.Sigmoid(),
                 idxTrainMovie = 0,
                 penaltyMultiplier = 0):

        super().__init__()
        self.idx = idxTrainMovie
        self.n_layers = len(depths)
        self.nonlinearity = nonlinearity
        self.term_idxs = [self.generate_term_idxs(len(GSOs),d) for d in depths]

        self.origGSOs = GSOs
        
        self.GSOs = [self.find_term_matrices(GSOs, self.term_idxs[layer]) for layer in range(len(self.term_idxs))]
        self.depths = depths
        self.penaltyMultiplier = penaltyMultiplier
        self.savedILconstant = None
        self.ILcounter = 0
        self.convLayers = []
        for i in range(len(depths)):
            f_in = fs[i]
            f_out = fs[i + 1]
            depth = self.depths[i]
            gfl = MultiGraphFilter(self.GSOs[i], f_in, f_out)
            activation = torch.nn.Sigmoid()
            self.convLayers += [gfl, activation]
            self.add_module(f"gfl{i}", gfl)
            self.add_module(f"activation{i}", activation)
        
        self.readoutLayers = []
        if len(readout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            readoutLayer1 = nn.Linear(fs[-1], readout[0])
            self.readoutLayers.append(readoutLayer1)
            self.add_module(f"rl{0}", readoutLayer1)
            # The last linear layer is not follow by a nonlinearity,
            # but is instead handled below in the forward function
            for l in range(len(readout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                activation = self.nonlinearity
                self.readoutLayers.append(activation)
                self.add_module(f"activationRL{l}", activation)
                # And add the linear layer
                readoutLayerL = nn.Linear(readout[l], readout[l+1])
                self.readoutLayers.append(readoutLayerL)
                self.add_module(f"rl{l+1}", readoutLayerL)
    
    def generate_term_idxs(self,numGSOs,depth):
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
          mat_term = GSOs[term[0]]
          if len(term) > 1:
            for j in range(1,len(term)):
              mat_term = mat_term @ GSOs[term[j]]
          coo = coo_matrix(mat_term)
          values = coo.data
          indices = np.vstack((coo.row, coo.col))

          i = torch.LongTensor(indices)
          v = torch.DoubleTensor(values)
          shape = coo.shape

          GSOsparse = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
          GSO_interactions.append(GSOsparse.transpose(0,1).to(device))
        return GSO_interactions
    
    def ILconstantComp(self, EOpers, layer, oper):
        ILconstantEstimations = []
        operator = self.origGSOs[oper]
        filter = self.convLayers[2 * layer].weight[:,0,:,:].cpu().detach().numpy()
        # Computes ILConstant for each layer, for each GSO

        for EO in EOpers:
            ILconstantEst = np.zeros((filter.shape[0],operator.shape[0],filter.shape[2],operator.shape[1]))
            for idx, term in enumerate(self.term_idxs[layer]):
                addTerm = np.zeros(EO.shape)
                for jdx, termItemI in enumerate(term):
                    if termItemI == oper:
                        if jdx == 0:
                            mat_term = EO
                            for kdx in range(1,len(term)):
                                mat_term = mat_term @ self.origGSOs[term[kdx]]
                        else:
                            mat_term = self.origGSOs[term[0]]
                            for kdx in range(1,len(term)):
                                if kdx == jdx:
                                    mat_term = mat_term @ EO
                                else:
                                    mat_term = mat_term @ self.origGSOs[term[kdx]]
                        addTerm += mat_term
                # Outer multiply filter by addTerm to get ILconstantEst for term
                ILconstantEst += np.einsum('ac,bd->abcd',filter[:,idx+1,:],addTerm)
            ILconstantEstimations.append(LA.norm(ILconstantEst))
        return max(ILconstantEstimations)

    def ILconstant(self, skipCalc = False):
        if skipCalc:
          return self.savedILconstant

        if self.savedILconstant is None or self.ILcounter == 7:
          self.ILcounter = 0
          ITERS = 5
          Es = []
          for i in range(ITERS):
            E = np.random.uniform(size=self.origGSOs[0].shape)
            Es.append(E / LA.norm(E))

          # Compute ILConstant for each layer, for each GSO
          # Should be size self.n_layers x len(self.origGSOs)
          ILconstants = np.zeros((self.n_layers, len(self.origGSOs)))
          for operator in range(len(self.origGSOs)):
            EOpers = [E @ self.origGSOs[operator] for E in Es]
            for layer in range(self.n_layers):
                  ILconstants[layer, operator] = self.ILconstantComp(EOpers, layer, operator)
          print(f'CALCULATED: {np.max(np.min(ILconstants, axis=1))}')
          # Find the minimum for each operator, and the maximum for each
          self.savedILconstant = np.max(np.min(ILconstants, axis=1))
        else:
          self.ILcounter += 1
        return self.savedILconstant
    

    def measureStability(self, errGSO, x):
        y1 = x.clone()
        y2 = x.clone()
        for i, layer in enumerate(self.convLayers):
            y1 = layer(y1, self.GSOs) if i % 2 == 0 else layer(y1)
        for i, layer in enumerate(self.convLayers):
            y2 = layer(y2, errGSO) if i % 2 == 0 else layer(y2)
        return LA.norm(y2 - y1)

    def forward(self, x):
        for i, layer in enumerate(self.convLayers):
            x = layer(x, self.GSOs) if i % 2 == 0 else layer(x)
        for i, layer in enumerate(self.readoutLayers):
            x = layer(x)
        # scale score to be between [1,5]
        return (1 + 4 * torch.sigmoid(x))[:,self.idx,:]