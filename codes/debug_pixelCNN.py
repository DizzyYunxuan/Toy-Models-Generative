import yaml
import torch
from autoregressive_blocks import PixelCNN
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2


num_input_c = 1
num_inner_c = 64
num_output_c = 256
num_masked_convs = 4
use_sigmoid = False
n_classes = 10
conditional = True
device = 'cuda'





model = PixelCNN(num_input_c=num_input_c, num_inner_c=num_inner_c, num_output_c=num_output_c, num_masked_convs=num_masked_convs, useSigmoid=use_sigmoid, num_classes=n_classes,conditional=conditional)

st_d = torch.load('/home/SENSETIME/weiyunxuan1/generative_models/expriments_save/20250305_train_AR_conPixCnn_CE/model_epoch_59_bak.pth')
st_d_new = {}
for k, v in st_d.items():
    new_k = k.replace('pixelCNN.', '')
    st_d_new[new_k] = v

model.load_state_dict(st_d_new)
model = model.to(device)



H, W = 28, 28
samples = torch.zeros(size=(1, 1, H, W)).to(device)
label = np.array([5])
label = F.one_hot(torch.tensor(label), num_classes=n_classes).to(device).float()
# with torch.no_grad():
#     for i in range(H):
#         for j in range(W):
#             if j > 0 and i > 0:
#                 out = model(samples, label)
#                 out_sm = torch.softmax(out[:, :, i, j], dim=1)
#                 samples[:, :, i, j] = torch.argmax(out_sm, dim=1, keepdim=True).float() / 255.0
#                 # print('i,j={}-{}='.format(i, j), samples[:, :, i, j])


#                 # samples[:, :, i, j] = torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

# samples = samples.cpu().numpy().transpose(0, 2, 3, 1) * 255
# plt.figure()
# plt.imshow(samples[0, :, :, :], 'gray')


samples = torch.zeros(size=(1, 1, H, W)).to(device)
samples[0, 0, ...] = torch.from_numpy(cv2.imread('/home/SENSETIME/weiyunxuan1/generative_models/test.png')[:,:,0] / 255.0) 
res = torch.zeros_like(samples)


for i in range(H):
    for j in range(W):
        out = model(samples, label)
        pred_sm = torch.softmax(out[:, :, i, j], dim=1)
        res[:, :, i, j] = torch.argmax(pred_sm, dim=1, keepdim=True) / 255.0
        print('i,j={}-{}='.format(i, j), samples[:, :, i, j])

res = res.cpu().numpy().transpose(0, 2, 3, 1) * 255
plt.figure()
plt.imshow(res[0, :, :, :], 'gray')
plt.figure()
plt.imshow(samples[0, 0, :, :].cpu().numpy(), 'gray')
plt.show()



