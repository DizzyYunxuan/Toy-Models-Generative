import torch
import numpy as np
from autoregressive_blocks import PixelCNN

class ResBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.convlRelu = [torch.nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
                        # torch.nn.LeakyReLU()
                        ]
        self.convlRelu = torch.nn.Sequential(*self.convlRelu)
    
    def forward(self, x):
        residual = self.convlRelu(x)
        return x + residual

class PixelCNN_Embedding(PixelCNN):
    def __init__(self, num_input_c=1, num_inner_c=64, num_output_c=64, num_masked_convs=4, useSigmoid=True, num_embedding=256, conditional=False, n_classes=10):
        # self.conditional = conditional
        # if self.conditional:
        #     num_input_c = num_input_c + 1
        super().__init__(num_input_c, num_inner_c, num_output_c, num_masked_convs, useSigmoid, num_classes=n_classes, conditional=conditional)

        # color_level = 256
        self.embedding = torch.nn.Embedding(num_embedding, num_inner_c)
        # if self.conditional:
        #     self.label_encoder = [torch.nn.Conv2d(in_channels=n_classes, out_channels=num_inner_c//2, kernel_size=3, stride=1, padding=1),
        #                           torch.nn.LeakyReLU(),
        #                           torch.nn.Conv2d(in_channels=num_inner_c//2, out_channels=1, kernel_size=3, stride=1, padding=1),
        #                           ]
        #     self.label_encoder = torch.nn.Sequential(*self.label_encoder)
    
    def forward(self, x, condition=None):

        # if self.conditional:
        #     b, h, w = x.shape
        #     condition = condition.unsqueeze(2)
        #     condition = condition.unsqueeze(3)
        #     condition = condition.repeat(1, 1, h, w)
        #     c = self.label_encoder(condition)
        # else:
        #     c = None


        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2)
        # x = torch.cat([x, c], dim=1)
        return super().forward(x, condition)



class VQVAE(torch.nn.Module):
    def __init__(self, input_channel, inner_channel, output_channel, num_embedding, n_classes=10, conditional=False):
        super().__init__()


        self.n_downsample = 4
        self.conditional = conditional

        self.encoder = [torch.nn.Conv2d(in_channels=input_channel, out_channels=inner_channel, kernel_size=3, stride=2, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, stride=2, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(),
                        ResBlock(dim=inner_channel),
                        ResBlock(dim=inner_channel)
                        ]
        self.embedding = torch.nn.Embedding(num_embedding, inner_channel)
        self.embedding.weight.data.uniform_(-1.0 / num_embedding,
                                            1.0 / num_embedding)

        self.decoder = [torch.nn.Conv2d(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(),
                        ResBlock(dim=inner_channel),
                        ResBlock(dim=inner_channel),
                        torch.nn.Conv2d(in_channels=inner_channel, out_channels=inner_channel * 2**2, kernel_size=3, stride=1, padding=1),
                        torch.nn.PixelShuffle(2),
                        torch.nn.Conv2d(in_channels=inner_channel, out_channels=inner_channel * 2**2, kernel_size=3, stride=1, padding=1),
                        torch.nn.LeakyReLU(),
                        torch.nn.PixelShuffle(2),
                        torch.nn.Conv2d(in_channels=inner_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
                        ]

        # if self.conditional:
        #     self.label_encoder = [torch.nn.Conv2d(in_channels=n_classes, out_channels=inner_channel//2, kernel_size=3, stride=1, padding=1),
        #                           torch.nn.LeakyReLU(),
        #                           torch.nn.Conv2d(in_channels=inner_channel//2, out_channels=1, kernel_size=3, stride=1, padding=1),
        #                         #   torch.nn.LeakyReLU()
        #                           ]
        #     self.label_encoder = torch.nn.Sequential(*self.label_encoder)

        self.encoder = torch.nn.Sequential(*self.encoder)
        self.decoder = torch.nn.Sequential(*self.decoder)


    def forward(self, x, condition=None):

        # if self.conditional:
        #     b, c, h, w = x.shape
        #     condition = condition.unsqueeze(2)
        #     condition = condition.unsqueeze(3)
        #     condition = condition.repeat(1, 1, h, w)
        #     c = self.label_encoder(condition)
        #     x = torch.cat([x, c], dim=1)

        # encoder
        ze = self.encoder(x)

        # embedding
        embedding = self.embedding.weight.data # [K, C]
        K, C = embedding.shape
        B, _, H, W = ze.shape

        embedding = embedding.reshape([1, K, C, 1, 1])
        ze_boardcast = ze.reshape([B, 1, C, H, W])

        distance = torch.sum((ze_boardcast - embedding) ** 2, 2) # [B, K, H, W]
        min_index = torch.argmin(distance, 1) # [B, H, W]
        zq = self.embedding(min_index) # [B, C, H, W]
        zq = zq.permute(0, 3, 1, 2)

        # decode
        decoder_input = ze + (zq - ze).detach()
        x_res = self.decoder(decoder_input)
        return ze, zq, x_res

    def encode(self, x):

        # encode
        ze = self.encoder(x)

        # embedding
        embedding = self.embedding.weight.data # [K, C]
        K, C = embedding.shape
        B, _, H, W = ze.shape

        embedding = embedding.reshape([1, K, C, 1, 1])
        ze = ze.reshape([B, 1, C, H, W])

        distance = torch.sum((ze - embedding) ** 2, 2) # [B, K, H, W]
        min_index = torch.argmin(distance, 1) # [B, H, W]
        return min_index
    
    def decode(self, min_index):
        zq = self.embedding(min_index).permute(0, 3, 1, 2)
        x_res = self.decoder(zq)
        return x_res


if __name__ == '__main__':
    device = 'cuda'
    vqvae = VQVAE(1, 64, 1, 32, conditional=True).to(device)
    test_sample = torch.ones([3, 1, 28, 28]).to(device)
    label = torch.randint(0, 10, [3, ]).to(device)
    label_oh = torch.nn.functional.one_hot(label, num_classes=10).float().to(device)

    r = vqvae(test_sample, label_oh)
    # print(r.shape)
