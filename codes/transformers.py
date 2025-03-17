import torch
from einops import repeat, rearrange
import cv2
from matplotlib import pyplot as plt
import numpy as np


class VisionTransformer(torch.nn.Module):
    def __init__(self, image_channel, image_size, patch_size, num_transformer, num_head, embed_size, num_class, classifierLayer=False) -> None:
        super().__init__()


        self.image_channel = image_channel
        self.num_patch = (image_size // patch_size) ** 2
        self.patch_size =  image_channel * patch_size ** 2
        self.classifierLayer = classifierLayer

        self.class_token = torch.nn.Parameter(torch.randn(1, 1, embed_size))
        self.positional_encoder = torch.nn.Parameter(torch.randn(1, self.num_patch + 1, embed_size))
        self.patch_encoder = torch.nn.Conv2d(image_channel, embed_size, patch_size, patch_size)
        # self.patch_encoder = torch.nn.Linear(self.patch_size, embed_size)

        trans_modules = []
        for i in range(num_transformer):
            trans_modules.append(TransformerEncoder(num_head=num_head, embed_size=embed_size))
        self.transformer_modules = torch.nn.Sequential(*trans_modules)
        
        if self.classifierLayer:
            self.classifier = torch.nn.Linear(embed_size, num_class)


    def forward(self, x):
        bs, c, h, w = x.shape # [bs, 1, 28, 28]
        x = self.patch_encoder(x) # [bs, embed_size, 4, 4]
        x = x.flatten(2) # [bs, embed_size, 16]
        x = x.transpose(1, 2) # [bs, 16, embed_size]

        class_token = repeat(self.class_token, '1 1 d -> b 1 d', b=bs ) # [bs, 1, 96]
        x = torch.cat([class_token, x], dim=1) # [bs, 17, 96]
        x = x + self.positional_encoder     # [bs, 17, 96]

        
        x = self.transformer_modules(x) # [bs, 16, 96]

        if self.classifierLayer:
            x = self.classifier(x) # [bs, 16, 10]
            x = torch.mean(x, dim=1) # [bs, 10]
        
        

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_head, embed_size) -> None:
        super().__init__()

        self.layer_norm_0 = torch.nn.LayerNorm(embed_size)

        self.multiHeadAtten = MultiHeadAttention(num_head=num_head, embed_size=embed_size)

        self.layer_norm_1 = torch.nn.LayerNorm(embed_size)
        self.mlp = [torch.nn.Linear(embed_size, embed_size * 4),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(embed_size * 4, embed_size)]
        self.mlp = torch.nn.Sequential(*self.mlp)


    def forward(self, x):
        x_res = self.layer_norm_0(x) # [bs, 17, 96]
        x_res = self.multiHeadAtten(x_res) + x

        x_res = self.layer_norm_1(x_res)
        x = self.mlp(x_res) + x_res
        return x
    


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_head, embed_size) -> None:
        super().__init__()

        self.Q = torch.nn.Linear(embed_size, embed_size)
        self.K = torch.nn.Linear(embed_size, embed_size)
        self.V = torch.nn.Linear(embed_size, embed_size)

        self.num_head = num_head
        self.head_size = embed_size // num_head
        assert self.head_size * num_head == embed_size
    
        self.softmax = torch.nn.Softmax(dim=-1)

        self.ffn = torch.nn.Linear(embed_size, embed_size)


    def forward(self, x):
        bs, l, emb_size = x.shape
        q = self.Q(x) # [bs, 17, 96]
        k = self.K(x) # [bs, 17, 96]
        v = self.V(x) # [bs, 17, 96]

        q = q.reshape(bs, l, self.num_head, self.head_size).permute(0, 2, 1, 3) # [bs, 3, 3, 17, 32]
        k = k.reshape(bs, l, self.num_head, self.head_size).permute(0, 2, 1, 3) # [bs, 3, 3, 17, 32]
        v = v.reshape(bs, l, self.num_head, self.head_size).permute(0, 2, 1, 3) # [bs, 3, 3, 17, 32]


        k = k.permute(0, 1, 3, 2) # [bs, 3, 3, 32, 17]
 
        qk_sqrtd = torch.matmul(q, k) * (self.head_size) ** (-0.5) # [bs, 3, 3, 17, 17]
        atten = torch.matmul(self.softmax(qk_sqrtd), v) # [bs, 3, 3, 17, 32]
        atten = atten.reshape(bs, l, emb_size) # [bs, 17, 96]
        atten = self.ffn(atten)  # [bs, 17, 96]

        return atten


if __name__ == '__main__':
    
    # test_tensor = torch.randn([3, 1, 28, 28])
    # vit = VisionTransformer(1, 28, 7, 3, 3, 96, 10)

    # print(test_tensor.shape)
    # res = vit(test_tensor)
    # print(res.shape)

    img = cv2.imread('./test.png')[:, :, 0] / 255.0
    img = torch.from_numpy(img).reshape(1, 1, 28, 28)
    print(img.shape)

    patches = img.reshape(1, 7 ** 2, 16) #(n,c,w,h) --> (n,7*7,16)

    res = np.zeros_like(img)[0, 0, :, :]
    for i in range(7**2):
        p = patches[0, i, :].reshape(4, 4)
        sh = i // 7
        sw = i % 7

        sh_index = sh * 4
        eh_index = (sh + 1) * 4

        sw_index = sw * 4
        ew_index = (sw + 1) * 4

        res[sh_index:eh_index, sw_index:ew_index] = p 

    print()


