import torch
from torch import nn
         
class ChannelAttention(nn.Module):
   def __init__(self, in_planes, ratio=16):
       super(ChannelAttention, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.max_pool = nn.AdaptiveMaxPool2d(1)
       self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
       self.relu1 = nn.ReLU()
       self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
       max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
       return self.sigmoid(avg_out + max_out)
         
class SpatialAttention(nn.Module):
   def __init__(self, kernel_size=7):
       super(SpatialAttention, self).__init__()
       self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size // 2), bias=False)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       avg_out = torch.mean(x, dim=1, keepdim=True)
       max_out, _ = torch.max(x, dim=1, keepdim=True)
       x = torch.cat([avg_out, max_out], dim=1)
       return self.sigmoid(self.conv1(x))
        
class CBAM(nn.Module):
   def __init__(self, in_planes, ratio=16, kernel_size=7):
       super(CBAM, self).__init__()
       self.ca = ChannelAttention(in_planes, ratio)
       self.sa = SpatialAttention(kernel_size)
   def forward(self, x):
       x = x * self.ca(x)
       return x * self.sa(x)
      
if __name__ == '__main__':
   cbam = CBAM(64)
   input_tensor = torch.rand(4, 64, 32, 32)
   output_tensor = cbam(input_tensor)
   print(input_tensor.size(), output_tensor.size())                                                        