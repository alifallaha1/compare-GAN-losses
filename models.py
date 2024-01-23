import torch 
import torch .nn as nn 


class Dicriminator(nn.Module):
    def __init__(self, channel ,filters):
        super().__init__()
        self.disc=nn.Sequential(## 64
            self.block(channel , filters , 4 , 2 , 1 ,False),## 32
            self.block(filters , filters*2 , 4 , 2 , 1),## 16 
            self.block(filters*2 , filters*4 , 4 , 2 , 1),## 8 
            self.block(filters*4 , filters*8 , 4 , 2 , 1),## 4 
            self.block(filters*8 , filters*16 , 4 , 2 , 1),##2
            nn.Conv2d(filters*16 , 1 , 4 ,2 ,1),##1
            nn.Sigmoid()

        )



    def block(self , in_ch , out_ch , kernel , stride , padding , batch=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)]
        if batch:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self,image):
        return self.disc(image)
    

class Genrator(nn.Module):
    def __init__(self, z_dim , channel , filters):
        super().__init__()
        self.gen=nn.Sequential(#1
            self.block(z_dim , filters*32 , 1 , 1 , 0),#1
            self.block(filters*32 , filters*16 , 4 , 2 , 1),#2
            self.block(filters*16 , filters*8 , 4 , 2 , 1),#4
            self.block(filters*8 , filters*4 , 4 , 2 , 1),#8
            self.block(filters*4 , filters*2 , 4 , 2 , 1),#16
            self.block(filters*2 , filters , 4 , 2 , 1),#32
            nn.ConvTranspose2d(filters , channel , 4 , 2 , 1),#64
            nn.Tanh()
        

        )


    def block(self , in_ch , out_ch , kernel , stride , padding):
        return nn.Sequential(
        nn.ConvTranspose2d(in_ch,out_ch,kernel,stride,padding , bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
        )
    
    def forward(self,noise):
        return self.gen(noise)
    


def init_weights(m):
    if isinstance(m,(nn.Conv2d , nn.ConvTranspose2d ,nn.BatchNorm2d )):
        nn.init.normal_(m.weight , mean=0 ,std=0.02)


def test():
    batch , z_dim , channels , f , size = 64 , 100 , 3 ,32 , 64
    image = torch.randn((batch , channels , size , size ))
    noise = torch.randn((batch , z_dim , 1 , 1))
    disc=Dicriminator(channels , f)
    init_weights(disc)
    gen = Genrator(z_dim , channels , f)
    init_weights(gen)
    disc(image)
    fake = gen(noise)
    disc(fake)
