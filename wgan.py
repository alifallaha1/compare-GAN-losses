import torch 
import torch .nn as nn 


class Critic(nn.Module):
    def __init__(self, channel ,filters):
        super().__init__()
        self.disc=nn.Sequential(## 64
            self.block(channel , filters , 4 , 2 , 1 ,False),## 32
            self.block(filters , filters*2 , 4 , 2 , 1),## 16 
            self.block(filters*2 , filters*4 , 4 , 2 , 1),## 8 
            self.block(filters*4 , filters*8 , 4 , 2 , 1),## 4 
            self.block(filters*8 , filters*16 , 4 , 2 , 1),##2
            nn.Conv2d(filters*16 , 1 , 4 ,2 ,1),##1

        )



    def block(self , in_ch , out_ch , kernel , stride , padding , batch=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)]
        if batch:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self,image):
        return self.disc(image)



class Critic_GP(nn.Module):
    def __init__(self, channel ,filters):
        super().__init__()
        self.disc=nn.Sequential(## 64
            self.block(channel , filters , 4 , 2 , 1 ,False),## 32
            self.block(filters , filters*2 , 4 , 2 , 1),## 16 
            self.block(filters*2 , filters*4 , 4 , 2 , 1),## 8 
            self.block(filters*4 , filters*8 , 4 , 2 , 1),## 4 
            self.block(filters*8 , filters*16 , 4 , 2 , 1),##2
            nn.Conv2d(filters*16 , 1 , 4 ,2 ,1),##1

        )



    def block(self , in_ch , out_ch , kernel , stride , padding , batch=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)]
        if batch:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self,image):
        return self.disc(image)


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

    




