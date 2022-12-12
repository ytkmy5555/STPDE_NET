from torch import nn
from sklearn.preprocessing import MinMaxScaler

class ConvNet(nn.Module):
    def __init__(self,input_dim, hidden_dim, kernel_size1,padding1,kernel_size2,padding2):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size1 = kernel_size1
        self.padding1 = padding1

        self.kernel_size2 = kernel_size2
        self.padding2 = padding2
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=self.input_dim ,
                              out_channels=16,
                              kernel_size=kernel_size1,
                              padding=self.padding1,
                              ))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=16,
                               out_channels=1,
                               kernel_size=kernel_size2,
                               padding=self.padding2,
                               ))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return  out
