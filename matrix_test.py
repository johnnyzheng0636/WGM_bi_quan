# Test algo with a single matrix
import torch
from torch import nn
import argparse
import pickle
from pathlib import Path

from my_util import load, quan, eval
from my_util.gen_mat import gen_saddle_matrix
from my_util.visualization import plot_mat_dist_bar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-m', "--matrix_test", type=int, default=0, help="run only matrix test with max n."
    )
    parser.add_argument(
        '-mg', "--matrix_test_max_group", type=int, default=131072, help="Max group."
    )
    parser.add_argument(
        '-fp', "--fig_dir", type=str, default="./fig", help="path to dir store fig and data plot the fig"
    )

    parser.add_argument(
        "--model", type=str, default='meta-llama/Llama-3.2-1B', help="model to load; for example `meta-llama/Llama-3.2-1B`."
    )
    parser.add_argument(
        "--layer_cache_dir", type=str, default="./hidden_data", help="directory to save the layer cache for experiments."
    )
    parser.add_argument(
        '-td', "--test_gen_matrix_dynamic", action='store_true', help="test dynamic grouping"
    )
    parser.add_argument(
        '-tg', "--test_gen_matrix_greedy", action='store_true', help="test greedy merging"
    )
    parser.add_argument(
        '-twg', "--test_gen_matrix_windowed_greedy", action='store_true', help="test windowed greedy merging"
    )
    parser.add_argument(
        '-g', "--graph_layer_0", action='store_true', help="Plot linear value frequency for linear in layer 0."
    )

    args = parser.parse_args()

    if args.matrix_test >= 2:
        eval_algo_matrix = eval.matrix_test(args.matrix_test, args.fig_dir)
        eval_algo_matrix.test()
    elif args.test_gen_matrix_dynamic or args.test_gen_matrix_greedy or args.test_gen_matrix_windowed_greedy:
        # test algo
        print('test algo start with gen matrix')
        
        # gen a matrix for test algorithm
        mat = gen_saddle_matrix((2, 5), right_mean=0.0, left_mean=-0.0, seed=2).matrix
        print(mat)
        sorted_abs, row_idx, col_idx, sorted_binary = quan.sort_abs(mat)
        cum_sum, cum_sq_sum = quan.prefix_sum(sorted_abs)

        if args.test_gen_matrix_dynamic:
            bi_mat = quan.dynamicGroupingQuan(
                mat, 
                lambda_reg=0,
                max_groups=2,
            )
        elif args.test_gen_matrix_greedy or args.test_gen_matrix_windowed_greedy:
            bi_mat = quan.greedyGroupingQuan(
                mat, 
                lambda_reg=0,
                max_groups=2,
                window=3,
            )

        print('sorted_abs: ', sorted_abs)
        print('sorted_binary: ', sorted_binary)
        print('prefix_sum: ', bi_mat.prefix_sum_abs)
        print('prefix_sq_sum: ', bi_mat.prefix_sum_sq_abs)

        if args.test_gen_matrix_dynamic or args.test_gen_matrix_greedy:
            bi_mat.grouping()
        else:
            bi_mat.window_grouping()            

    else:
        # load from pickle linear chunk and use one weigth matrix for test
        if args.graph_layer_0:  # plot frequency dist of value of linear of layer 0
            chunk_file = 'layer_0_linear.pickle'
            chunk_path = Path(args.layer_cache_dir) / args.model.replace('/', '__') / chunk_file
            with open(chunk_path, 'rb') as fs:
                chunk = pickle.load(fs)
            children = chunk
            fig_dir = Path('./fig') / args.model.replace('/', '__')
            fig_dir.mkdir(parents=True, exist_ok=True)
            for child in children:
                tmpPath = fig_dir / f'{child}.png'
                print(child)
                print(children[child])
                plot_mat_dist_bar(children[child].weight, tmpPath)
        
        # test algo
        print('test algo start with real weight')