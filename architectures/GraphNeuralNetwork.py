class GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 GSOs,
                 ks: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1),
                 readout: Union[List[int], Tuple[int]] = (1, 1),
                 f_edge = 1,
                 nonlinearity = torch.nn.Sigmoid(),
                 idxTrainMovie = 0,
                 penaltyMultiplier = 0):
        """
        An L-layer graph neural network. Uses Sigmoid activation for each layer

        Args:
            ks: [K_1,...,K_L]. On ith layer, K_{i} is the number of filter taps.
            fs: [F_1,...,F_L]. On ith layer, F_{i} and F_{i+1} are the number of input and output features, respectively.
            readouts: [F_1,...,F_L]. On ith layer, F_{i} and F_{i+1} are the number of input and output features, respectively.
        """
        super().__init__()
        deviceGSOs = []
        N = GSOs[0].shape[0]
        self.eigenvalues = np.zeros((len(GSOs), N))
        
        for ind, GSO in enumerate(GSOs):
          w, _ = LA.eig(GSO)
          self.eigenvalues[ind,:] = w
          coo = coo_matrix(GSO)
          values = coo.data
          indices = np.vstack((coo.row, coo.col))

          i = torch.LongTensor(indices)
          v = torch.DoubleTensor(values)
          shape = coo.shape

          GSOsparse = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
          deviceGSOs.append(GSOsparse.to(device))
        
        thisvK = np.ones((f_edge,N))
        vK = thisvK.reshape(f_edge, 1, N)
        for k in range(1,max(ks)):
            thisvK = thisvK * self.eigenvalues
            vK = np.concatenate((vK, thisvK.reshape(f_edge, 1, N)), axis = 1)
        self.eigenvalues = torch.tensor(self.eigenvalues).to(device)
        self.eigenvaluesPowers = torch.tensor(vK).to(device)
        self.F = fs
        self.K = ks

        self.S = deviceGSOs
        self.idx = idxTrainMovie
        self.n_layers = len(ks)
        self.nonlinearity = nonlinearity
        self.convLayers = []
        self.f_edge = f_edge
        self.penaltyMultiplier = penaltyMultiplier
        for i in range(len(ks)):
            f_in = fs[i]
            f_out = fs[i + 1]
            k = ks[i]
            gfl = GraphFilter(k, f_in, f_out, self.f_edge)
            activation = self.nonlinearity
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
            x = layer(x, self.S) if i % 2 == 0 else layer(x)
        for i, layer in enumerate(self.readoutLayers):
            x = layer(x)
        # scale score to be between [1,5]
        return (1 + 4 * torch.sigmoid(x))[:,self.idx,:]

    def ILconstant(self, skipCalc = False):
        
        E = self.f_edge
        N = self.S[0].shape[1]
        
        # Let's move onto each parameter
        l = 0 # Layer counter
        ILconstant = torch.empty(0).to(self.S[0].device)
            # Initial value for the IL penalty
        # For each parameter,
        for param in self.parameters():
            # Check it has dimension 4 (it is the filter taps)
            if len(param.shape) == 4:
                # Check if the dimensions coincide, the param has to have
                # Fl x E x Kl x Gl
                # where Fl, Gl and Kl change with each layer l, but E is fixed
                assert param.shape[0] == self.F[l+1] # F
                assert param.shape[1] == E           # E
                assert param.shape[2] == self.K[l]   # K
                assert param.shape[3] == self.F[l]   # G
                Fl = param.shape[0]
                Kl = param.shape[2]
                Gl = param.shape[3]
                # We will do elementwise multiplication between the filter taps
                # and the eigenvalue, and then sum up through the k dimension
                # to compute the derivative.
                # The derivative is sum_{k=1}^{K-1} k h_{k} lambda^{k-1} so we
                # need to start counting on k=1 for the filter taps, and go only
                # up to all but the last element for lambda.
                
                # So first, get rid of the first element along the k dimension
                # of the filters
                param = torch.narrow(param, 2, 1, Kl-1)
                #   Fl x E x (Kl-1) x Gl
                #   We use narrow because it shares the same storage, so it
                #   doesn't overwhelm the GPU memory
                # We adapt it to multiplication with the eigenvalues
                param = param.reshape([Fl, E, Kl-1, Gl, 1])
                #   Fl x E x (Kl-1) x Gl x 1
                # Repeat it the length of the eigenvalues
                param = param.repeat([1, 1, 1, 1, N])
                #   Fl x E x (Kl-1) x Gl x N
                
                # Now we prepare the eigenvalue powers
                eigPowers = self.eigenvaluesPowers[:, 0:Kl-1, :]
                #   E x (Kl-1) x N
                # We just got all but the last eigenvalue power
                eigPowers = eigPowers.reshape(1, E, Kl-1, 1, N)
                #   1 x E x (Kl-1) x 1 x N
                # So now it's ready and in shape for multiplication
                
                # And finally we need the value of k that increased from 1 to
                # Kl-1
                kLinear = torch.arange(1, Kl,
                                       device = param.device, dtype = param.dtype)
                kLinear = kLinear.reshape(1, 1, Kl-1, 1, 1)
                #   1 x 1 x (Kl-1) x 1 x 1
                
                # And now we are ready to build the derivative
                hPrime = kLinear * param * eigPowers
                #   h'(lambda): Derivative of h
                #   Fl x E x (Kl-1) x Gl x N
                # And add up along the k dimension
                hPrime = torch.sum(hPrime, dim = 2)
                #   Fl x E x Gl x N
                
                # Now, we need to multiply h'(lambda) with lambda.
                # The eigenvalues have shape E x N
                # hPrime has shape Fl x E x Gl x N
                # So we need to reshape the eigenvalues
                thisILconstant = self.eigenvalues.reshape([1, E, 1, N]) * hPrime
                #   Fl x E x Gl x N
                
                # Apply torch.abs and torch.max over the nSamples dimension
                # (second argument are the positions of the maximum, which we 
                # don't care about).
                thisILconstant, _ = torch.max(torch.abs(thisILconstant), dim=3)
                #   Fl x E x Gl
                # This torch.max does not have a second argument because it is
                # the maximum of all the numbers
                thisILconstant = torch.max(thisILconstant)
                # Add the constant to the list
                ILconstant = torch.cat((ILconstant,thisILconstant.unsqueeze(0)))
                # And increase the number of layers
                l = l + 1
        
        # After we computed the IL constant for each layer, we pick the
        # maximum and go with that
        return torch.max(ILconstant)