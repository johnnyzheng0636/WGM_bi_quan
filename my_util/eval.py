# ppl function and matrix test
import torch
from torch import nn
from . import quan
import time
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

import subprocess

def eval_qa(model, outputPath="./evalQA"):

    outputPath = Path(outputPath) / model
    outputPath.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "lm_eval", 
        "--model hf \\", 
        f"--model_args pretrained={model},trust_remote_code=True \\", 
        "--tasks piqa,boolq,openbookqa,winogrande,arc_easy,arc_challenge,hellaswag \\", 
        "--device cuda:0 \\", 
        "--log_samples \\", 
        f"--output_path {outputPath} \\", 
        "--batch_size auto \\", 
        "--trust_remote_code "])

@torch.no_grad()
def ppl(model, testenc, dev, ):
    print("Evaluating ...")
    model.seqlen = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    # update to transformer 4.45
    # add this new layer
    model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # transformer 4.45 required position_ids instead of attention mask for Llama
    pos_ids = torch.zeros(
        (nsamples, 1, model.seqlen), dtype=torch.int16, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            pos_ids[cache["i"]] = kwargs["position_ids"]
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), position_ids=pos_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache



class matrix_test():
    """
    Give the matrix loss for different quantization method,
    Should be experiemnted with different n(matrix of shape (n, n))
    """
    def __init__(self, n, fig_dir):
        self.n = n
        self.loss_fn = torch.nn.MSELoss()

        
        self.saveDir = Path(fig_dir) / "matrix_test"
        self.saveDir.mkdir(parents=True, exist_ok=True)

        # n to speed and loss
        self.n_to_speed_loss_dict = {}

        self.n_to_loss_path_fig = self.saveDir / f"{n}n_to_loss.png"
        self.n_to_speed_path_fig = self.saveDir / f"{n}n_to_speed.png"

        self.n_to_speed_loss_path_csv = self.saveDir / f"{n}n_to_speed_loss.csv"

        # exponential n to speed and loss
        self.exp_n_to_speed_loss_dict = {}

        self.exp_n_to_loss_path_fig = self.saveDir / f"{2**n}n_to_loss.png"
        self.exp_n_to_speed_path_fig = self.saveDir / f"{2**n}n_to_speed.png"

        self.exp_n_to_speed_loss_path_csv = self.saveDir / f"{2**n}n_to_speed_loss.csv"

        # lambda to loss
        self.lambda_to_loss_dict = {}

        self.lambda_to_loss_path_fig = self.saveDir / f"lambda_to_loss.png"

        self.lambda_to_loss_path_csv = self.saveDir / f"lambda_to_loss.csv"

        # max groups to speed and loss
        # using 512**2 matrix for speed
        # max group set to half of total elements, i.e. 512**2/2
        self.max_groups_to_speed_loss_dict = {}

        self.max_groups_to_loss_path_fig = self.saveDir / f"{131072}g_to_loss.png"
        self.max_groups_to_speed_path_fig = self.saveDir / f"{131072}g_to_speed.png"
        self.max_groups_vs_avg_bit = self.saveDir / f"{131072}g_vs_avg_bit.png"

        self.max_groups_to_speed_loss_path_csv = self.saveDir / f"{131072}g_to_speed_loss.csv"

        # window to speed and loss
        self.window_to_speed_loss_dict = {}

        self.window_to_loss_path_fig = self.saveDir / f"{2048}w_to_loss.png"
        self.window_to_speed_path_fig = self.saveDir / f"{2048}w_to_speed.png"

        self.window_to_speed_loss_path_csv = self.saveDir / f"{2048}w_to_speed_loss.csv"

    def n_to_speed_loss(self, ):
        '''
        Figure 2 and 4 in appendix C
        '''
        if not os.path.exists(self.n_to_speed_loss_path_csv):
            for i in range(2, self.n+1):
                print('current n: ', i)
                # gen a matrix of shape (n, n)

                # test with different quantization methods

                # how loss and speed change as n increase
                test_shape = (i, i)
                test_w = torch.normal(0, 1, size=test_shape)

                # dummy baseline
                # print('dummy')
                dummy_w = torch.zeros(test_shape)
                loss_dummy = self.loss_fn(dummy_w, test_w)

                dummy_dict = {'time': -1, 'mse': loss_dummy}

                del dummy_w
                del loss_dummy

                # plain xnor
                # print("no_grouping")
                start_t = time.time()
                binarize_xnor = quan.Binarization(test_w, group_method='no')
                test_w_quan_xnor= binarize_xnor.quantization()
                loss_xnor = self.loss_fn(test_w_quan_xnor, test_w)

                xnor_dict = {'time': time.time()-start_t, 'mse': loss_xnor}
                
                del loss_xnor
                del binarize_xnor
                del test_w_quan_xnor

                # column wise block xnor
                # print("quan_k_col_block")
                start_t = time.time()
                col_wise_binarize = quan.Binarization(test_w, group_method='col_wise_block', col_block_len=256)
                test_quan_k_col_block = col_wise_binarize.quantization()
                loss_k_col_block = self.loss_fn(test_quan_k_col_block, test_w)

                col_blk_xnor_dict = {'time': time.time()-start_t, 'mse': loss_k_col_block}

                del col_wise_binarize
                del test_quan_k_col_block
                del loss_k_col_block

                # TODO
                
                # dynamic_grouping_quan
                print("dynamic_grouping_quan")
                start_t = time.time()
                dynamic_grouping_quan = quan.dynamicGroupingQuan(
                            test_w, 
                            lambda_reg=0.75,
                            max_groups=32,
                        )
                dynamic_grouping_quan.grouping()
                loss_dynamic_grouping = self.loss_fn(dynamic_grouping_quan.quantized_simu, test_w)

                dynamic_grouping_dict = {'time': time.time()-start_t, 'mse': loss_dynamic_grouping}

                del dynamic_grouping_quan
                del loss_dynamic_grouping

                print("greddy_grouping_quan")
                start_t = time.time()
                greddy_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=0.75,
                            max_groups=32,
                        )
                greddy_grouping_quan.grouping()
                loss_greedy_grouping = self.loss_fn(greddy_grouping_quan.quantized_simu, test_w)

                greedy_grouping_dict = {'time': time.time()-start_t, 'mse': loss_greedy_grouping}

                del greddy_grouping_quan
                del loss_greedy_grouping

                # greddy_win_grouping_quan
                print("greddy_win_grouping_quan")
                start_t = time.time()
                greddy_win_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=0.75,
                            max_groups=32,
                            window=512,
                            # lambda_nor=True
                        )
                greddy_win_grouping_quan.window_grouping()
                loss_win_greedy_grouping = self.loss_fn(greddy_win_grouping_quan.quantized_simu, test_w)

                greddy_win_dict = {'time': time.time()-start_t, 'mse': loss_win_greedy_grouping}

                del greddy_win_grouping_quan
                del loss_win_greedy_grouping

                self.n_to_speed_loss_dict[i] = {
                    'dummy': dummy_dict,
                    'xnor': xnor_dict,
                    'col_blk_xnor': col_blk_xnor_dict,
                    'dynamic_group': dynamic_grouping_dict,
                    'greedy_group': greedy_grouping_dict,
                    'greddy_win': greddy_win_dict,
                }
                # print(self.n_to_speed_loss_dict[i])

            # print(self.n_to_speed_loss_dict)
            # Flatten the nested data into rows
            rows = []
            for group in self.n_to_speed_loss_dict:
                for method in self.n_to_speed_loss_dict[group]:
                    time_val = self.n_to_speed_loss_dict[group][method]['time']
                    mse_val = self.n_to_speed_loss_dict[group][method]['mse'].item()  # Extract scalar from tensor
                    rows.append({
                        'group': group,
                        'method': method,
                        'time': time_val,
                        'mse': mse_val
                    })

            # Write to CSV
            with open(self.n_to_speed_loss_path_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['group', 'method', 'time', 'mse'])
                writer.writeheader()
                writer.writerows(rows)

            # Extract unique methods (excluding 'dummy' for time plot)
            methods = {row['method'] for row in rows}
            groups = sorted(self.n_to_speed_loss_dict.keys())

            # Time Plot (Log Scale)
            plt.figure(figsize=(12, 6))
            for method in methods:
                if method == 'dummy':
                    continue  # Skip dummy for time plot
                times = [row['time'] for row in rows if row['method'] == method]
                plt.plot(groups, times, marker='o', label=method)

            plt.xlabel('n')
            plt.ylabel('Time (seconds)')
            plt.yscale('log')
            plt.title('Time Comparison (Log Scale)')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.n_to_speed_path_fig)
            plt.close()

            # MSE Plot
            plt.figure(figsize=(12, 6))
            for method in methods:
                mses = [row['mse'] for row in rows if row['method'] == method]
                plt.plot(groups, mses, marker='o', label=method)

            plt.xlabel('n')
            plt.ylabel('MSE')
            plt.title('MSE Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.n_to_loss_path_fig)
            plt.close()
        else:
            # Load CSV
            df = pd.read_csv(self.n_to_speed_loss_path_csv)
            
            # Pivot the DataFrame for plotting
            pivot_df = df.pivot(index='group', columns='method', values='mse')

            # Plot
            pivot_df.plot(marker='o', figsize=(12, 6))
            plt.xlabel('n')
            plt.ylabel('MSE')
            plt.title('MSE Comparison')
            plt.grid(True)
            plt.legend(title='Method')
            plt.savefig(self.n_to_loss_path_fig )
            plt.close()
            
            # Pivot the DataFrame for plotting
            pivot_df = df.pivot(index='group', columns='method', values='time')

            # Plot
            pivot_df.plot(marker='o', figsize=(12, 6))
            plt.xlabel('n')
            plt.ylabel('Time (seconds)')
            plt.yscale('log')
            plt.title('Time Comparison (Log Scale)')
            plt.grid(True)
            plt.legend(title='Method')
            plt.savefig(self.n_to_speed_path_fig )
            plt.close()

    def exp_n_to_speed_loss(self, ):
        '''
        figure 3 and 5 in Appendix C
        '''
        if not os.path.exists(self.exp_n_to_speed_loss_path_csv):
            exp_n = [2**i for i in range(1, self.n+1)]
            for i in exp_n:
                print('current n: ', i)
                # gen a matrix of shape (n, n)

                # test with different quantization methods

                # how loss and speed change as n increase
                test_shape = (i, i)
                test_w = torch.normal(0, 1, size=test_shape)

                # dummy baseline
                # print('dummy')
                dummy_w = torch.zeros(test_shape)
                loss_dummy = self.loss_fn(dummy_w, test_w)

                dummy_dict = {'time': -1, 'mse': loss_dummy}

                del dummy_w
                del loss_dummy

                # plain xnor
                # print("no_grouping")
                start_t = time.time()
                binarize_xnor = quan.Binarization(test_w, group_method='no')
                test_w_quan_xnor= binarize_xnor.quantization()
                loss_xnor = self.loss_fn(test_w_quan_xnor, test_w)

                xnor_dict = {'time': time.time()-start_t, 'mse': loss_xnor}
                
                del loss_xnor
                del binarize_xnor
                del test_w_quan_xnor

                # column wise block xnor
                # print("quan_k_col_block")
                start_t = time.time()
                col_wise_binarize = quan.Binarization(test_w, group_method='col_wise_block', col_block_len=256)
                test_quan_k_col_block = col_wise_binarize.quantization()
                loss_k_col_block = self.loss_fn(test_quan_k_col_block, test_w)

                col_blk_xnor_dict = {'time': time.time()-start_t, 'mse': loss_k_col_block}

                del col_wise_binarize
                del test_quan_k_col_block
                del loss_k_col_block

                # TODO
                
                # dynamic_grouping_quan
                # print("dynamic_grouping_quan")
                # start_t = time.time()
                # dynamic_grouping_quan = quan.dynamicGroupingQuan(
                #             test_w, 
                #             lambda_reg=0.75,
                #             max_groups=32,
                #         )
                # dynamic_grouping_quan.grouping()
                # loss_dynamic_grouping = self.loss_fn(dynamic_grouping_quan.quantized_simu, test_w)

                # dynamic_grouping_dict = {'time': time.time()-start_t, 'mse': loss_dynamic_grouping}

                # del dynamic_grouping_quan
                # del loss_dynamic_grouping

                print("greddy_grouping_quan")
                start_t = time.time()
                greddy_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=0.75,
                            max_groups=32,
                        )
                greddy_grouping_quan.grouping()
                loss_greedy_grouping = self.loss_fn(greddy_grouping_quan.quantized_simu, test_w)

                greedy_grouping_dict = {'time': time.time()-start_t, 'mse': loss_greedy_grouping}

                del greddy_grouping_quan
                del loss_greedy_grouping

                # greddy_win_grouping_quan
                print("greddy_win_grouping_quan")
                start_t = time.time()
                greddy_win_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=0.75,
                            max_groups=32,
                            window=512,
                            # lambda_nor=True
                        )
                greddy_win_grouping_quan.window_grouping()
                loss_win_greedy_grouping = self.loss_fn(greddy_win_grouping_quan.quantized_simu, test_w)

                greddy_win_dict = {'time': time.time()-start_t, 'mse': loss_win_greedy_grouping}

                del greddy_win_grouping_quan
                del loss_win_greedy_grouping

                self.exp_n_to_speed_loss_dict[i] = {
                    'dummy': dummy_dict,
                    'xnor': xnor_dict,
                    'col_blk_xnor': col_blk_xnor_dict,
                    # 'dynamic_group': dynamic_grouping_dict,
                    'greedy_group': greedy_grouping_dict,
                    'greddy_win': greddy_win_dict,
                }
                # print(self.exp_n_to_speed_loss_dict[i])

            # print(self.exp_n_to_speed_loss_dict)
            # Flatten the nested data into rows
            rows = []
            for group in self.exp_n_to_speed_loss_dict:
                for method in self.exp_n_to_speed_loss_dict[group]:
                    time_val = self.exp_n_to_speed_loss_dict[group][method]['time']
                    mse_val = self.exp_n_to_speed_loss_dict[group][method]['mse'].item()  # Extract scalar from tensor
                    rows.append({
                        'group': group,
                        'method': method,
                        'time': time_val,
                        'mse': mse_val
                    })

            # Write to CSV
            with open(self.exp_n_to_speed_loss_path_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['group', 'method', 'time', 'mse'])
                writer.writeheader()
                writer.writerows(rows)

            # Extract unique methods (excluding 'dummy' for time plot)
            methods = {row['method'] for row in rows}
            groups = sorted(self.exp_n_to_speed_loss_dict.keys())

            # Time Plot (Log Scale)
            plt.figure(figsize=(12, 6))
            for method in methods:
                if method == 'dummy':
                    continue  # Skip dummy for time plot
                times = [row['time'] for row in rows if row['method'] == method]
                plt.plot(groups, times, marker='o', label=method)

            plt.xlabel('n')
            plt.ylabel('Time (seconds)')
            plt.yscale('log')
            plt.title('Time Comparison (Log Scale)')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_n_to_speed_path_fig)
            plt.close()

            # MSE Plot
            plt.figure(figsize=(12, 6))
            for method in methods:
                mses = [row['mse'] for row in rows if row['method'] == method]
                plt.plot(groups, mses, marker='o', label=method)

            plt.xlabel('n')
            plt.ylabel('MSE')
            plt.title('MSE Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_n_to_loss_path_fig)
            plt.close()
        else:
            # Load CSV
            df = pd.read_csv(self.exp_n_to_speed_loss_path_csv)
            
            # Pivot the DataFrame for plotting
            pivot_df = df.pivot(index='group', columns='method', values='mse')

            # Plot
            pivot_df.plot(marker='o', figsize=(12, 6))
            plt.xlabel('n')
            plt.ylabel('MSE')
            plt.title('MSE Comparison')
            plt.grid(True)
            plt.legend(title='Method')
            plt.savefig(self.exp_n_to_loss_path_fig)
            plt.close()
            
            # Pivot the DataFrame for plotting
            pivot_df = df.pivot(index='group', columns='method', values='time')

            # Plot
            pivot_df.plot(marker='o', figsize=(12, 6))
            plt.xlabel('n')
            plt.ylabel('Time (seconds)')
            plt.yscale('log')
            plt.title('Time Comparison (Log Scale)')
            plt.grid(True)
            plt.legend(title='Method')
            plt.savefig(self.exp_n_to_speed_path_fig)
            plt.close()

    def lambda_to_loss(self, ):
        '''
        Figure 6 in Appendix C
        '''
        if not os.path.exists(self.lambda_to_loss_path_csv):
            # how loss change as lambda change for fixed n
            # test only greedy mergin and windowed greedy
            # how loss change as lambda increase
            i = 512
            test_shape = (i, i)
            test_w = torch.normal(0, 1, size=test_shape)

            # dummy baseline
            # print('dummy')
            dummy_w = torch.zeros(test_shape)
            loss_dummy = self.loss_fn(dummy_w, test_w)

            dummy_dict = {'mse': loss_dummy}
            for l in range(0, 101, 5):
                l = l/100
                print('current lambda: ', l)
                print("greddy_grouping_quan")
                greddy_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=l,
                            max_groups=32,
                            lambda_nor=True,
                        )
                greddy_grouping_quan.grouping()
                loss_greedy_grouping = self.loss_fn(greddy_grouping_quan.quantized_simu, test_w)

                greedy_grouping_dict = {'mse': loss_greedy_grouping}

                del greddy_grouping_quan
                del loss_greedy_grouping

                # greddy_win_grouping_quan
                print("greddy_win_grouping_quan")
                greddy_win_grouping_quan = quan.greedyGroupingQuan(
                            test_w, 
                            lambda_reg=l,
                            max_groups=32,
                            window=128,
                            lambda_nor=True,
                        )
                greddy_win_grouping_quan.window_grouping()
                loss_win_greedy_grouping = self.loss_fn(greddy_win_grouping_quan.quantized_simu, test_w)

                greddy_win_dict = {'mse': loss_win_greedy_grouping}

                del greddy_win_grouping_quan
                del loss_win_greedy_grouping

                self.lambda_to_loss_dict[l] = {
                    'dummy': dummy_dict,
                    'greedy_group': greedy_grouping_dict,
                    'greddy_win': greddy_win_dict,
                }

                rows = []
                for group in sorted(self.lambda_to_loss_dict.keys()):  # Process groups in order
                    for method in self.lambda_to_loss_dict[group]:
                        mse_val = self.lambda_to_loss_dict[group][method]['mse'].item()  # Extract scalar from tensor
                        rows.append({
                            'group': group,
                            'method': method,
                            'mse': mse_val
                        })

                # Write to CSV
                with open(self.lambda_to_loss_path_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['group', 'method', 'mse'])
                    writer.writeheader()
                    writer.writerows(rows)

                # Extract unique methods and sorted groups
                methods = sorted({row['method'] for row in rows})  # Alphabetically sorted methods
                groups = sorted(self.lambda_to_loss_dict.keys())  # X-axis values (sorted groups)

                # Plot MSE
                plt.figure(figsize=(12, 6))
                for method in methods:
                    # Get MSE values for this method in group order
                    mses = [row['mse'] for row in rows if row['method'] == method]
                    plt.plot(groups, mses, marker='o', linestyle='-', label=method)

                plt.xlabel('lambda')
                plt.ylabel('MSE (Loss)')
                plt.title('MSE Comparison for different Lambda')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.lambda_to_loss_path_fig)
                plt.close()
        else:
            # Load CSV
            df = pd.read_csv(self.lambda_to_loss_path_csv)
            df = df[df.method != 'dummy']
            
            # Pivot the DataFrame for plotting
            pivot_df = df.pivot(index='group', columns='method', values='mse')
            # Plot
            pivot_df.plot(marker='o', figsize=(12, 6))
            plt.xlabel('Lambda')
            plt.ylabel('MSE')
            plt.title('MSE Comparison')
            plt.grid(True)
            plt.legend(title='Method')
            plt.savefig(self.lambda_to_loss_path_fig)
            plt.close()



    def max_groups_to_speed_loss(self, ):
        '''
        Figure 7, 8, and 9 in Appendix C
        '''
        # how loss and speed change as max_groups change for fixed n
        # test only greedy merging and windowed greedy
        i = 512
        total_n = 512**2
        exp_n = [2**i for i in range(5, 17+1)]
        # average bit for different max_group
        avg_bit = [1 + i*16/total_n for i in exp_n]

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot with markers and line
        plt.plot(exp_n, avg_bit, 'b-o', markersize=6, linewidth=1.5)

        # Set logarithmic scale for x-axis (since groups are powers of 2)
        plt.xscale('log')
        plt.xticks(exp_n, labels=[str(x) for x in exp_n], rotation=45)

        

        # Add text annotations for each point
        for x, y in zip(exp_n, avg_bit):
            plt.text(
                x, 
                y * 1.005,  # Slightly offset above the point
                f"{y:.5f}".rstrip('0').rstrip('.') if '.' in f"{y:.5f}" else f"{y:.5f}",
                ha='center',
                va='bottom',
                fontsize=9,
                rotation=45
            )


        # Formatting
        plt.xlabel('Group Size', fontsize=12)
        plt.ylabel('Average bit length', fontsize=12)
        plt.title('Average bit length vs Group Size', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.max_groups_vs_avg_bit)

        # gen a matrix of shape (n, n)

        # test with different quantization methods

        # how loss and speed change as n increase
        test_shape = (i, i)
        test_w = torch.normal(0, 1, size=test_shape)

        for i in exp_n:
            print('current max groups: ', i)
            print("greddy_grouping_quan")
            start_t = time.time()
            greddy_grouping_quan = quan.greedyGroupingQuan(
                        test_w, 
                        lambda_reg=0.75,
                        max_groups=i,
                    )
            greddy_grouping_quan.grouping()
            loss_greedy_grouping = self.loss_fn(greddy_grouping_quan.quantized_simu, test_w)

            greedy_grouping_dict = {'time': time.time()-start_t, 'mse': loss_greedy_grouping}

            del greddy_grouping_quan
            del loss_greedy_grouping

            # greddy_win_grouping_quan
            print("greddy_win_grouping_quan")
            start_t = time.time()
            greddy_win_grouping_quan = quan.greedyGroupingQuan(
                        test_w, 
                        lambda_reg=0.75,
                        max_groups=i,
                        window=512,
                        # lambda_nor=True
                    )
            greddy_win_grouping_quan.window_grouping()
            loss_win_greedy_grouping = self.loss_fn(greddy_win_grouping_quan.quantized_simu, test_w)

            greddy_win_dict = {'time': time.time()-start_t, 'mse': loss_win_greedy_grouping}

            del greddy_win_grouping_quan
            del loss_win_greedy_grouping

            self.max_groups_to_speed_loss_dict[i] = {
                'greedy_group': greedy_grouping_dict,
                'greddy_win': greddy_win_dict,
            }
            # print(self.max_groups_to_speed_loss_dict[i])

        # print(self.max_groups_to_speed_loss_dict)
        # Flatten the nested data into rows
        rows = []
        for group in self.max_groups_to_speed_loss_dict:
            for method in self.max_groups_to_speed_loss_dict[group]:
                time_val = self.max_groups_to_speed_loss_dict[group][method]['time']
                mse_val = self.max_groups_to_speed_loss_dict[group][method]['mse'].item()  # Extract scalar from tensor
                rows.append({
                    'group': group,
                    'method': method,
                    'time': time_val,
                    'mse': mse_val
                })

        # Write to CSV
        with open(self.max_groups_to_speed_loss_path_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['group', 'method', 'time', 'mse'])
            writer.writeheader()
            writer.writerows(rows)

        # Extract unique methods (excluding 'dummy' for time plot)
        methods = {row['method'] for row in rows}
        groups = sorted(self.max_groups_to_speed_loss_dict.keys())

        # Time Plot (Log Scale)
        plt.figure(figsize=(12, 6))
        for method in methods:
            if method == 'dummy':
                continue  # Skip dummy for time plot
            times = [row['time'] for row in rows if row['method'] == method]
            plt.plot(groups, times, marker='o', label=method)

        plt.xlabel('Group')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        plt.title('Time Comparison (Log Scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.max_groups_to_speed_path_fig)
        plt.close()

        # MSE Plot
        plt.figure(figsize=(12, 6))
        for method in methods:
            mses = [row['mse'] for row in rows if row['method'] == method]
            plt.plot(groups, mses, marker='o', label=method)

        plt.xlabel('Group')
        plt.ylabel('MSE')
        plt.title('MSE Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.max_groups_to_loss_path_fig)
        plt.close()


    def window_to_speed_loss(self, ):
        '''
        Figure 10 and 11 in Appendix C
        '''
        # how loss and speed change as window change for fixed n
        # test only windowed greedy
        exp_n = [2**i for i in range(1, 11+1)]
        # gen a matrix of shape (n, n)

        # test with different quantization methods

        # how loss and speed change as n increase
        i = 512
        test_shape = (i, i)
        test_w = torch.normal(0, 1, size=test_shape)

        for i in exp_n:
            # greddy_win_grouping_quan
            print("greddy_win_grouping_quan")
            start_t = time.time()
            greddy_win_grouping_quan = quan.greedyGroupingQuan(
                        test_w, 
                        lambda_reg=0.75,
                        max_groups=64,
                        window=i,
                        # lambda_nor=True
                    )
            greddy_win_grouping_quan.window_grouping()
            loss_win_greedy_grouping = self.loss_fn(greddy_win_grouping_quan.quantized_simu, test_w)

            greddy_win_dict = {'time': time.time()-start_t, 'mse': loss_win_greedy_grouping}

            del greddy_win_grouping_quan
            del loss_win_greedy_grouping

            self.window_to_speed_loss_dict[i] = {
                'greddy_win': greddy_win_dict,
            }
            # print(self.window_to_speed_loss_dict[i])

        # print(self.window_to_speed_loss_dict)
        # Flatten the nested data into rows
        rows = []
        for group in self.window_to_speed_loss_dict:
            for method in self.window_to_speed_loss_dict[group]:
                time_val = self.window_to_speed_loss_dict[group][method]['time']
                mse_val = self.window_to_speed_loss_dict[group][method]['mse'].item()  # Extract scalar from tensor
                rows.append({
                    'group': group,
                    'method': method,
                    'time': time_val,
                    'mse': mse_val
                })

        # Write to CSV
        with open(self.window_to_speed_loss_path_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['group', 'method', 'time', 'mse'])
            writer.writeheader()
            writer.writerows(rows)

        # Extract unique methods (excluding 'dummy' for time plot)
        methods = {row['method'] for row in rows}
        groups = sorted(self.window_to_speed_loss_dict.keys())

        # Time Plot (Log Scale)
        plt.figure(figsize=(12, 6))
        for method in methods:
            if method == 'dummy':
                continue  # Skip dummy for time plot
            times = [row['time'] for row in rows if row['method'] == method]
            plt.plot(groups, times, marker='o', label=method)

        plt.xlabel('Window size')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        plt.title('Time Comparison (Log Scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.window_to_speed_path_fig)
        plt.close()

        # MSE Plot
        plt.figure(figsize=(12, 6))
        for method in methods:
            mses = [row['mse'] for row in rows if row['method'] == method]
            plt.plot(groups, mses, marker='o', label=method)

        plt.xlabel('Window size')
        plt.ylabel('MSE')
        plt.title('MSE Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.window_to_loss_path_fig)
        plt.close()


    def test(self, ):
        # plot corresponding graph
        self.n_to_speed_loss()
        self.exp_n_to_speed_loss()
        self.max_groups_to_speed_loss()
        self.lambda_to_loss()
        self.window_to_speed_loss()
        