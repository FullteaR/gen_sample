import numpy as np
import PIL.Image
import torch
import pickle
import sys

path = 'network.pkl'
with open(path, 'rb') as f:
    G = pickle.load(f)['G_ema']

if len(sys.argv)==1:
    seed = np.random.randint(0,2**31)
else:
    seed = int(sys.argv[1])
z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
c = 0

w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save('test.png')