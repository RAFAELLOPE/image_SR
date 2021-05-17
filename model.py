import torch.nn as nn
import torch

class DCSRNSubblock(nn.Module):
  def __init__(self, in_chans, out_chans):
    super(DCSRNSubblock, self).__init__()
    self.batchnorm = nn.BatchNorm3d(in_chans)
    self.celu = nn.CELU()
    self.conv = nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.conv(self.celu(self.batchnorm(x)))
    return x



class DCSRN(nn.Module):
  def __init__(self, n_chans):
    super(DCSRN, self).__init__()
    
    self.initial_block = nn.Conv3d(1, 2*n_chans, kernel_size=3, padding=1)

    self.inter_block = DCSRNSubblock(in_chans=2*n_chans,
                                     out_chans=n_chans)
    
    self.final_block =  nn.Conv3d(2*n_chans, 1, kernel_size=3, padding=1)


  def forward(self, x):
    x_in1 = self.initial_block(x)

    x_out1 = self.inter_block(x_in1)
    x_out1 = torch.cat((x_out1, x_out1), dim=1)
    x_in2 = x_out1 + x_in1

    x_out2 = self.inter_block(x_in2)
    x_out2 = torch.cat((x_out2, x_out2), dim=1)
    x_in3 = x_out2 + x_in2

    x_out3 = self.inter_block(x_in3)
    x_out3 = torch.cat((x_out3, x_out3), dim=1)
    x_in4 = x_out3 + x_in3

    x_out4 = self.inter_block(x_in4)
    x_out4 = torch.cat((x_out4, x_out4), dim=1)
    x_in5 = x_out4 + x_in4

    return self.final_block(x_in5)