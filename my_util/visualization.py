# plot graph
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_mat_dist_bar(mat, savePath):
    """
    Plot the distribution of input matrix as bar chart
    
    Args:
        mat (torch.Tensor): input matrix
    """
    # expect to be a saddle like shape with 0 begin the saddle
    plt.clf()
    # flatten
    arr = torch.flatten(mat).detach().numpy()

    # plot frequenct again value
    counts, bins = np.histogram(arr, bins='auto')
    plt.hist(arr, bins=bins, edgecolor='black', alpha=0.5)
    plt.title('Frequency Distribution (Histogram)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(savePath)