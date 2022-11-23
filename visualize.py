import os
import glob
import numpy as np
import pandas as pd  # pandas v1.3.5
from tqdm import tqdm 
import pickle
import torch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from warnings import filterwarnings
filterwarnings("ignore")


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    return data


def make_plot(data: list, save_path):
    
    # df = pd.DataFrame()
    dvecs_list = []
    for data in d_vectors:
        dvecs = data['d_vectors']
        # dvecs = np.array(dvecs.to('cpu').detach(), dtype=object)
        # if dvecs.size(0) == 1:
        #     dvecs = torch.cat([dvecs, dvecs], dim=0)

        # dvecs = dvecs.detach().to('cpu').cpu().numpy()
        dvecs_list.append(dvecs.detach().to('cpu'))
        # print(dvecs.shape)
        
        # break
    dvecs_list = np.array(torch.cat(dvecs_list, dim=0), dtype=object)

    # dvecs_list = np.array(dvecs_list, dtype=object)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=725)
    transformed = tsne.fit_transform(dvecs_list)

    # df = pd.DataFrame({"spk_id" : [d_vectors[i]['spk_id'] for i in range(len(d_vectors))],
    #                     "dim_X" : transformed[:, 0],
    #                     "dim_Y" : transformed[:, 1]})
 
    data = {
            "dim_X": transformed[:, 0],
            "dim_Y": transformed[:, 1],
            "label": [d_vectors[i]['spk_id'] for i in range(len(d_vectors))],
            }
    # print(f'data type:{type(data)}')

    plt.figure()
    sns.scatterplot(
        x="dim_X",
        y="dim_Y",
        hue="label",
        palette=sns.color_palette(n_colors=len(df.spk_id.unique())),
        data=data,
        legend="full",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path, format='png')


if __name__ == '__main__':
    save_path = 'vectors.png'

    d_vectors = load_pickle("d_vectors.pickle")  # [{spk_id, dvecs}]
    make_plot(d_vectors, save_path)



    