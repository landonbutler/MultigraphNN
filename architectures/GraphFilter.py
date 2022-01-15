



class GraphFilter(nn.Module):
    def __init__(self, k: int, f_in=1, f_out=1, f_edge=1, bias=True, sparse=False):
        """
        A graph filter layer.
        Args:
            gso: The graph shift operator.
            k: The number of filter taps.
            f_in: The number of input features.
            f_out: The number of output features.
        """
        super().__init__()
        self.k = k
        self.f_in = f_in
        self.f_out = f_out
        self.f_edge = f_edge
        self.sparse = sparse

        self.weight = nn.Parameter(torch.ones(self.f_out, self.f_edge, self.k, self.f_in))
        torch.nn.init.normal_(self.weight, 0, 3)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.f_out, 1)) 
            torch.nn.init.normal_(self.bias, 0, 3)

    def forward(self, x: torch.Tensor, S: list):
        batch_size = x.shape[0]
        assert len(S) == self.f_edge
        B = batch_size
        E = self.f_edge
        F = self.f_out
        G = self.f_in
        N = S[0].shape[-1]  # number of nodes
        K = self.k   # number of filter taps

        h = self.weight
        b = self.bias

        # Now, we have x in B x G x N and S in E x N x N, and we want to come up
        # with matrix multiplication that yields z = x * S with shape
        # B x E x K x G x N
        # For this, we first add the corresponding dimensions
        x = torch.unsqueeze(x,1).permute(0,1,3,2)
        z = torch.unsqueeze(x,2).repeat(1, E, 1, 1, 1) 
        # We need to repeat along the E dimension, because for k=0, S_{e} = I for
        # all e, and therefore, the same signal values have to be used along all
        # edge feature dimensions.
        for k in range(1, K):
            
            # Sparse multiplication
            xNew = torch.zeros(B,E,G,N).to(device)
            for batch in range(B):
                for e in range(E):
                    # e = 0 for x since it is same signal
                    xNew[batch,e,:,:] = (S[e].transpose(0,1) @ x[batch,0,:,:].T).T
            x = xNew
            xS = x.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
            del xNew
            z = torch.cat((z, xS), dim=2)  # B x E x k x G x N}
            del xS
            # torch.cuda.empty_cache()

        # This output z is of size B x E x K x G x N
        # Now we have the x*S_{e}^{k} product, and we need to multiply with the
        # filter taps.
        # We multiply z on the left, and h on the right, the output is to be
        # B x N x F (the multiplication is not along the N dimension), so we reshape
        # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
        # always reshape the last dimensions), and then make h be E x K x G x F and
        # reshape it to EKG x F, and then multiply
        y = torch.matmul(z.permute(0,4,1,2,3).reshape([B, N, E * K * G]), h.reshape([E * K * G, F])).permute(0,2,1)
        # y = z.squeeze().reshape(B,G,N)
        # And permute again to bring it from B x N x F to B x F x N.
        # Finally, add the bias
        if b is not None:
            y = y + b
        return y.permute(0,2,1)