#!/usr/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv, exit
from glob import glob


try:
  infolder1 = argv[1]
  ignore1 = int(argv[2])  # ignore the first labels.
  infolder2 = argv[3]
  ignore2 = int(argv[4])  # ignore the first labels.
  plotfile = argv[5]
except IndexError:
  print('Usage: python hist_per_label.py infolder ignore1 infolder ignore2 plotfile.png')
  exit()

dfs = []
for infolder, ignore in [(infolder1, ignore1), (infolder2, ignore2), ('../../data/preprocessed_datasets/brats12', 0)]:
  for infile in glob(infolder + '/*.npz')[:10]:
    img = np.load(infile)

    data = img['data'][..., [0]]  # assuming t1 is the first modality
    seg = img['seg']
    seg[seg <= max(0, ignore)] = ignore
    seg -= ignore

    dataset = infolder.split('/')[-1]  # this is pretty ad hoc
    if dataset=='brats12' or dataset=='brainweb': dataset = 'brainweb + brats12'
    tmp_df = pd.DataFrame()

    tmp_df['value'] = data[seg != 0].flatten()
    tmp_df['label'] = seg[seg != 0].flat
    tmp_df['dataset'] = dataset

    dfs.append(tmp_df)

df = pd.concat(dfs)
# for label in range(1, seg.max()+1):
#   sns.kdeplot(t1_data[seg == label].reshape(-1), label=label, shade=True, bw='silverman')
# plt.legend()
# plt.savefig(plotfile)
# g = sns.JointGrid(x='label', y='value', data=df[df['dataset'] == 'tumorsim'])
# g = g.plot_marginals(sns.kdeplot, shade=True)
# g = sns.JointGrid(x='label', y='value', data=df[df['dataset'] == 'brainweb'])
# g = g.plot_marginals(sns.kdeplot, shade=True)
# # g = g.plot_joint(sns.violinplot, palette='muted', split=True)
# ax = g.ax_joint
# ax.axis('off')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# inset_axes = inset_axes(ax,
#                         width="100%", # width = 30% of parent_bbox
#                         height="100%", # height : 1 inch
#                         loc=10,
#                         borderpad=0)
sns.set()
sns.violinplot(x='label', y='value', data=df, hue='dataset', palette='muted', split=True)#, ax=inset_axes)
# plt.legend()
plt.savefig(plotfile)
plt.clf()
for dataset in df['dataset'].unique():
  for label in df['label'].unique():
    sns.kdeplot(df[(df['label'] == label) & (df['dataset'] == dataset)]['value'], label=label, shade=True, bw='silverman')
  plt.savefig(plotfile[:-4] + dataset + '.png')
  plt.clf()