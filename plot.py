import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--module_idx', default='1', type=str)
parser.add_argument('--scale', default='true', type=str)
args = parser.parse_args()

def plot():
  for epoch in ['0', '9', '29', '59']:
    for method in ['ngd', 'exact', 'kfac']:
      try:
        with open(method + '/' + epoch + '_m_' + args.module_idx + '_inv.npy', 'rb') as f:
          inv = np.load(f)
        if args.scale == 'true':
          inv = np.log(np.abs(inv) + 1e-3)

        fig, ax = plt.subplots(figsize=(18,18))
        im = ax.imshow(inv, cmap='coolwarm', vmin=np.min(inv), vmax=np.max(inv))
        fig.colorbar(im, orientation='horizontal')
        plt.show()
        fig.savefig(method + '/' + epoch + '_m_' + args.module_idx + '_inv.png')
      except FileNotFoundError:
        print('[Error] missing: epoch = ' + epoch + ' method = ' + method + ' for module ' + args.module_idx)

if __name__ == '__main__':
  plot()