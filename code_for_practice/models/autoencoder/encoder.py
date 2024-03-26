import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# 이미지를 가져와 넣으면 작은 사이즈로 변환시켜줌.
class VAE_Encoder(nn.Sequential):
  def __init__(self):
    super.__init__(
      # (batch_size, channel, h, w) -> (b, 128, h, w)
      nn.Conv2d(3, 128, kernel_size=3, padding=1),
      VAE_ResidualBlock(128,128),
      VAE_ResidualBlock(128,128),

      # (b, 128, h, w) -> (b, 128, h/2, w/2)
      nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
      VAE_ResidualBlock(128,256),
      VAE_ResidualBlock(256,256),

      # (b, 256, h/2, w/2) -> (b, 256, h/4, w/4)
      nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
      VAE_ResidualBlock(256,512),
      VAE_ResidualBlock(512,512),

      # (b, 512, h/4, w/4) -> (b, 512, h/8, w/8)
      nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
      VAE_ResidualBlock(512,512),
      VAE_ResidualBlock(512,512),

      VAE_ResidualBlock(512,512),
      VAE_AttentionBlock(512),
      VAE_ResidualBlock(512,512),

      nn.GroupNorm(32,512),

      nn.SiLU(),

      # (b, 512, h/4, w/4) -> (b, 8, h/8, w/8)
      nn.Conv2d(512,8, kernel_size=3, padding=1),

      # (b, 8, h/8, w/8)
      nn.Conv2d(8,8,kernel_size=1, padding=0),


    )

  def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

    for module in self:
      #stride가 (2,2)일 때만 추가적으로 아래쪽, 왼쪽에 padding을 넣어줌.
      if getattr(module, 'stride', None) == (2, 2):
        x = F.pad(x, (0,1,0,1))
      x = module(x)

    # (b, 4, h/8, w/8)
    # torch.chunk (input, chunks, dim=0)
    # chunk 개수로 쪼개줌. => 2개씩 나누어줌.
    mean, log_variance = torch.chunk(x, 2, dim=1)

    # torch.clamp (input, min, max)
    # min, man 사이의 값으로 조정해주는 것
    log_variance = torch.clamp(log_variance, -30, 20) #log 분산
    variance = log_variance.exp() # 분산
    stdev = variance.sqrt() # 표준편차

    # 자신의 분산, 자신의 평균을 가지고 noise를 삽입한 x를 제작함.
    # 이거로 가우시안 분포로 데이터를 변환하는 것을 의미 
    # Variational AutoEncoder에 대해 자세히 공부할 것
    x = mean + stdev * noise # noise 삽입

    x *= 0.18215 # 상수배로 scaling -> CompVis/stable-diffusion 에 있는 상수

    return x


