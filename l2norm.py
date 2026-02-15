import torch 
import torch.nn as nn
import torch.nn.init as init

class L2norm(nn.Module):
    def __init__(self,nb_channels,scale):
        """  https://arxiv.org/pdf/1506.04579 introduces L2 normalisation
        
        in ssd article https://arxiv.org/pdf/1512.02325 they use layer normalisation across channels
        initialisation to 20 
        """
        super().__init__()
        self.scale=scale
        self.nb_channels=nb_channels
        
        self.gamma=nn.Parameter(torch.empty(nb_channels))
        self.init_parameters()
        

    def init_parameters(self):
        init.constant_(self.gamma,self.scale)

    def forward(self,X):

        #sanity checked : normalise over channels for each N,H,W , since [N,C,H,W] is X tensor shape
        norm=torch.norm(X,dim=1,keepdim=True)
        x_hat=X/norm

        y_scaled=x_hat*self.gamma.view(1,self.nb_channels,1,1)
        #pytorch recall : to multiply across channels tensors of shape [N,C,H,W] and [C], simply reshape second to [1,C,1,1]

        return y_scaled