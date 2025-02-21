import torch



one = torch.ones([1, 1, 8, 8])
two = torch.ones([1, 1, 8, 8]) * 2
thr = torch.ones([1, 1, 8, 8]) * 3

cat = torch.cat([one, two, thr], dim=0)

print(cat[0, 0, :, :])

cat_rs = cat.reshape([3, 4, 4, 4])

print(cat_rs[0, 0, :, :])


