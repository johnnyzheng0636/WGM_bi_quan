# load model and dataset, then quantize and cache weights

import traceback
from torch import nn
import torch
from pathlib import Path
from my_util import quan
import pickle
import os
import glob
import time
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from . import eval
import concurrent.futures
import psutil

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        # LlamaTokenizer only support Llama 2
        if "meta-llama__Llama-3.2-1B".lower() in model.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=False)
        elif "meta-llama__Llama-3.2-3B".lower() in model.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    print('getting traindata')
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=True
    )
    print('got traindata, getting val data')
    # add trust_remote_code=True due to datasets 3.0.1 updates
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', trust_remote_code=True
    )
    print('got val data, tokenizing')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # print('i', i)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            # print('trainenc', trainenc)
            if trainenc.input_ids.shape[1] > seqlen:
                # print('trainenc.input_ids.shape[1]', trainenc.input_ids.shape[1])
                # print('seqlen', seqlen)
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    cache_file=f'datacache/{name}_{nsamples}_{seed}_{seqlen}_{model}.pt'
    print('pwd: ', os.getcwd())
    print('except cache at: ', cache_file)
    try:
        print('attempt loading cache')
        return torch.load(cache_file)
    except:
        pass
    print('no cache found, proceed to download')

    tokenizer = get_tokenizer(model)
    
    if 'wikitext2' in name:
        loaders= get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        loaders= get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        loaders= get_c4(nsamples, seed, seqlen, model, tokenizer)
    directory='/'.join(cache_file.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(loaders,cache_file)
    return loaders

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def load_model(model):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    print(model)

    return model

def save_layer(model, save_path):
    # load model
    saveDir = Path(save_path) / model.replace('/', '__')
    saveDir.mkdir(parents=True, exist_ok=True)

    print(saveDir)

    model = load_model(model)

    print(model)

    # get all linear layers
    layers = model.model.layers

    # print(layers)

    print('='*50)
    for i in range(len(layers)):
        layer = layers[i]
        children = find_layers(layer)
        print(children)
        print('+'*50)
        for child in children:
            print(child)
            print(children[child])
            # print(children[child].weight)
            # print(f'{child} shape: ',children[child].weight.shape)
            # print(children[child].bias)
            # try:
            #     print(children[child].bias.shape)
            # except:
            #     print('no bias')
            print('='*50)
        # save each layer as pickle for small chunk experiments
        tmp_path = saveDir / f'layer_{i}_linear.pickle'
        with open(tmp_path, 'wb') as fs:
            pickle.dump(children, fs, protocol=pickle.HIGHEST_PROTOCOL)


class binaryQuantization():
    def __init__(
        self,
        model_name = "meta-llama/Llama-3.2-1B", 
        save_dir = "./hidden_data",
        quanMethod = "windowed_greedy",     # only using the windowed greedy merging algorithm, since other are too slow
        lambda_reg=0.75,
        max_groups=32,
        window=512,
        loss_fn=nn.MSELoss(reduction='sum'),
        overwrite_quan=False,
        seed=0,
        evalonly=False,
        chunk=0,
        avgBit=1.08,
    ):
        """
        load the model, then apply the quantization method choosen on the model
        and save the quantized model

        Weight of the original linear layer and after quantization is saved independently 
        for analysis and CUDA kernel
        """
        self.avgBit = avgBit
        self.chunk = chunk
        self.evalonly = evalonly
        self.seed = seed
        self.overwrite_quan = overwrite_quan
        self.model_name = model_name
        self.save_dir = save_dir
        self.lambda_reg = lambda_reg
        self.max_groups = max_groups
        self.window = window
        self.quanMethod = quanMethod
        self.loss_fn = loss_fn

        
        print('mem at the start of init')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

        # savePath
        if not self.evalonly:
            if self.avgBit is not None:
                self.savesubdir = model_name.replace('/', '__') + f"_{self.quanMethod}_{self.avgBit}b_{self.window}w_{self.lambda_reg}l"
            else:
                self.savesubdir = model_name.replace('/', '__') + f"_{self.quanMethod}_{self.max_groups}g_{self.window}w_{self.lambda_reg}l"
            self.savePath_ori = Path(save_dir) / self.savesubdir / 'original_linear'
            self.savePath_ori.mkdir(parents=True, exist_ok=True)
            self.savePath_meta = Path(save_dir) / self.savesubdir / 'quantized_linear'
            self.savePath_meta.mkdir(parents=True, exist_ok=True)
            self.savePath_quanmodel = Path(save_dir) / self.savesubdir / self.savesubdir
            
            self.savePath_mp_tmp_ori = Path(save_dir) / 'mp_tmp' / 'original'
            self.savePath_mp_tmp_qtz = Path(save_dir) / 'mp_tmp' / 'quantized'
            self.savePath_mp_tmp_ori.mkdir(parents=True, exist_ok=True)
            self.savePath_mp_tmp_qtz.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        print('mem after path create of init')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")
        
        self.model = load_model(model_name)
        
        print('mem after load model of init')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

        # the algorithm is slower on gpu since it have no matrix multiplication
        # and relis on the depending while loop without CUDA adaption
        # self.model.to(self.device)

        # if "gemma-3" in self.model_name.lower():
        self.layers = self.model.model.layers
        
        print('mem after getting att layer of init')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

        self.linear_childern = {}

        for i in range(len(self.layers)):
            layer = self.layers[i]
            children = find_layers(layer)
            self.linear_childern[i] = children

        
        print('mem after find linear of model of init')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

        # find using code from paper "THE SUPER WEIGHT IN LARGE LANGUAGE MODELS"
        # hard coded using the result for each model
        # paper's
        # self.superWeight = {
        #     'meta-llama/Llama-3.2-3B': {
        #         1: {'mlp.down_proj': (588, 1419)},
        #         27: {'mlp.down_proj': (1016, 424)}
        #     }
        # }
        # reserve coor just in case, I got it wrong
        self.superWeight = {
            'meta-llama/Llama-3.2-3B': {
                1: {'mlp.down_proj': (1419, 588)},
                27: {'mlp.down_proj': (424, 1016)}
            }
        }
        # self.superWeight = {}

    # TODO Add store local then transfer to scratch if cpu not connected to scratch

    def save_layer(self, ):
        """
        Same as the function out of this class
        """
        print("saving linear layers in model")
        for i in self.linear_childern.keys():
            tmp_path = self.savePath_ori / f'layer_{i}_linear.pickle'
            if not os.path.exists(tmp_path):
                print(f"Saving layer{i}: \n", self.linear_childern[i])
                with open(tmp_path, 'wb') as fs:
                    pickle.dump(self.linear_childern[i], fs, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"{tmp_path} exists, skipped")

    def quan(self, weight):
        """
        perform the actual quantization
        """
        quantized_w = None
        quantized_bi = None
        quantized_scale = None
        quantized_scale_meta = None
        if self.avgBit is None:
            max_groups = self.max_groups
        else:
            tot_ele = weight.shape[0] * weight.shape[1]
            max_groups = (self.avgBit-1) * tot_ele // 16
        print(max_groups, "max groups for quantization")
        if self.quanMethod == "windowed_greedy":
            quan_instance = quan.greedyGroupingQuan(
                weight, 
                lambda_reg=self.lambda_reg,
                max_groups=max_groups,
                window=self.window,
            )
            quan_instance.window_grouping()
            
            quantized_w = quan_instance.quantized_simu
            quantized_bi = quan_instance.quantized_bi
            quantized_scale = quan_instance.quantized_scale
            quantized_scale_meta = quan_instance.quantized_scale_meta

        elif self.quanMethod == "blocked_xnor":
            quan_instance = quan.Binarization(
                weight, 
                group_method='col_wise_block', 
                col_block_len=self.window,
            )
            quantized_w = quan_instance.quantization()

        elif self.quanMethod == "xnor":
            quan_instance = quan.Binarization(
                weight, 
                group_method='no', 
                col_block_len=self.window,
            )
            quantized_w = quan_instance.quantization()
        else:
            return None
        return quantized_w, quantized_bi, quantized_scale, quantized_scale_meta


    def quantize_layer(self, layer):
        start = time.time()
        quantized_linear_childern_meta = {}
        quan_info = ""
        tmp_path = self.savePath_meta / f'layer_{layer}_linear.pickle'
        if not os.path.exists(tmp_path) or self.overwrite_quan:
            print("just before layer quantization start")
            for linear in self.linear_childern[layer].keys():
                now = time.time()
                print(f"layer {layer}, {linear} start quantization at {now - start:.2f}s")
                quantized_w, quantized_bi, quantized_scale, quantized_scale_meta = self.quan(
                    self.linear_childern[layer][linear].weight
                )
                print(f"layer {layer}, {linear} quantization done at {now - start:.2f}s")

                quantized_linear_childern_meta[linear] = {
                    'bi': quantized_bi,
                    'scale': quantized_scale,
                    'scale_meta': quantized_scale_meta,
                }
                loss = self.loss_fn(quantized_w, self.linear_childern[layer][linear].weight)
                quan_info += f"layer {layer}, {linear} loss: {loss}\n"
                # print(f"layer {layer}, {linear} loss: {loss}")
                self.linear_childern[layer][linear].weight = nn.Parameter(quantized_w)
            
            print(f"layer {layer} all quantization done at {now - start:.2f}s")
            with open(tmp_path, 'wb') as fs:
                pickle.dump(quantized_linear_childern_meta, fs, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"layer {layer} quantization saved at {now - start:.2f}s")
            quan_info += f"Quantizing layer {layer} done with {time.time() - start} seconds\n"
        else:
            with open(tmp_path, 'rb') as f:
                quantized_linear_childern_meta = pickle.load(f)
                for linear in self.linear_childern[layer].keys():
                    quantized_w = quantized_linear_childern_meta[linear]['bi'] * quantized_linear_childern_meta[linear]['scale']
                    self.linear_childern[layer][linear].weight = nn.Parameter(quantized_w)
            quan_info += f"{tmp_path} exists/skip flag (-oq) on, skipped\n"

        return quan_info


    # parallel version 
    # Per layer may is not speeding up, may due to layer is too large
    # TODO try per weight instead of per layer

    # worker foo for run_mp_per_weight
    def quantize_weight(self, root, file, startTime):
        """
        load and quantize the weight pickle
        worker function for run_mp_per_weight
        """
        tmpWeightPath = Path(root) / file
        pklWeightPath = self.savePath_mp_tmp_qtz / file
        quanInfo = {}

        print(f"wieght: {tmpWeightPath}")
        with open(tmpWeightPath, 'rb') as f:
            tmpWeight = pickle.load(f)
            
            now = time.time()
            # print(f"{file} start quantization at {now - startTime:.2f}s")
            quantized_w, quantized_bi, quantized_scale, quantized_scale_meta = self.quan(
                tmpWeight
            )
            print(f"{file} quantization done at {now - startTime:.2f}s")

            quanInfo = {
                'w': quantized_w,
                'bi': quantized_bi,
                'scale': quantized_scale,
                'scale_meta': quantized_scale_meta,
            }
            loss = self.loss_fn(quantized_w, tmpWeight)
            quanInfo['loss'] = loss
            print(f"{file} loss: {loss}")

        with open(pklWeightPath, 'wb') as f:
            pickle.dump(quanInfo, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")


    # worker for reconstruct
    def reconstruct(self, rc_path, rc_linear_childern):
        if os.path.exists(rc_path):
            with open(rc_path, 'rb') as f:
                quanInfo = pickle.load(f)
                # print(quanInfo)
                # print(quanInfo['bi'])
                # print(quanInfo['scale'])
                # print(quanInfo['scale_meta'])
                # print('weight in rc layer: \n', children[weight].weight)
                # print('weight in rc layer shape: ', children[weight].weight.shape)
                # assign the quantized weight to the model
                rc_linear_childern[rc_path].weight = nn.Parameter(
                    # quanInfo['bi'] * quanInfo['scale']
                    quanInfo['w']  # use this if you want to use the quantized weight directly
                )
        else:
            print(f"Quantized weight {rc_path} not found, skip this weight")
        
        print(f'mem after load {rc_path} into model')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

    @torch.no_grad()
    def run_mp_per_weight(self, ):
        """
        This paralle attempt will run weight by weight

        Since whole model occupied too much memory, this method will first load
        the model into memory, then save the linear weights to pickle files in an 
        original dir, next memory is emptied and load each linear weight from the 
        original dir and quantize it and save to another quantized dir.

        Finally the use the weight saved in the quantized dir to constructure the 
        quantized model and save the whole model as pt file.

        Again suffer from hardware limitation. Instead of one job, spilt this into
        8 job (max job) to see see.
        """
        startTime = time.time()
        
        print("mem at start of mp")
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")
        
        # self.savePath_mp_tmp_ori.mkdir(parents=True, exist_ok=True)
        # self.savePath_mp_tmp_qtz.mkdir(parents=True, exist_ok=True)

        # save all linear wieght to pickle files in the original dir
        if self.chunk < 1:
            for i in self.linear_childern.keys():
                # print(i)
                # print(self.linear_childern[i])
                for weight in self.linear_childern[i].keys():
                    tmp_path = self.savePath_mp_tmp_ori / f'layer_{i}_{weight}.pickle'
                    # print(f"Saving layer {i}, {weight} to {tmp_path}")
                    with open(tmp_path, 'wb') as fs:
                        pickle.dump(self.linear_childern[i][weight].weight, fs, protocol=pickle.HIGHEST_PROTOCOL)

            print('mem after saving original linear weights')
            print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

            # empty the model from memory
            del self.model
            del self.layers
            del self.linear_childern
            print('mem after delete model from mem')
            print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

        # parallel quantization for each linear weight from original dir
        # and save to quantized dir
        # TODO skip this after quantization done, test reconstruct the model
        # then unmask this and substitute with worker func for parallel test
        # serial is correct, test paralle next for correctness and speed up

        # parallel version
        # for root, _, files in os.walk(self.savePath_mp_tmp_ori, topdown=False):
        #     # max_cpu per job fixed by slurm
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #         # Submit tasks to the pool and store future objects
        #         future_to_weightQuan = {executor.submit(self.quantize_weight, root, file, startTime): file for file in files}
                
        #         # Process completed tasks as they finish
        #         for future in concurrent.futures.as_completed(future_to_weightQuan):
        #             file = future_to_weightQuan[future]
        #             nowTime = time.time()
        #             try:
        #                 # quan_info = future.result()
        #                 print(f"weight {file} quantized at {nowTime - startTime:.2f}")
        #             except Exception as e:
        #                 print(f"weight {file} quantization generated an exception: {str(e)}")

        #         break

        # serial version
        for root, _, files in os.walk(self.savePath_mp_tmp_ori, topdown=False):
            files = sorted(files)  # sort files to ensure consistent order
            print(files)
            # The for loop of walk only have 1 iteration, i.e. it is not a loop
            if self.chunk != 0 or self.chunk != 9:
                chunk_start = len(files) * (self.chunk - 1) // 8
                chunk_end = len(files) * self.chunk // 8
                files = files[chunk_start:chunk_end]
                print(chunk_start, chunk_end)
                print("current chunk: ", files)
            for file in files:
                tmpWeightPath = Path(root) / file
                pklWeightPath = self.savePath_mp_tmp_qtz / file

                # load and quantize the weight pickle, upgrade this to a worker func
                quanInfo = {}

                print(f"wieght: {tmpWeightPath}")
                with open(tmpWeightPath, 'rb') as f:
                    tmpWeight = pickle.load(f)
                    
                    now = time.time()
                    print(f"{file} start quantization at {now - startTime:.2f}s")
                    quantized_w, quantized_bi, quantized_scale, quantized_scale_meta = self.quan(
                        tmpWeight
                    )
                    print(f"{file} quantization done at {now - startTime:.2f}s")

                    # 4-bit meta information is too much so only store the quantized weight
                    if self.avgBit == None:
                        quanInfo = {
                            'w': quantized_w,
                            'bi': quantized_bi,
                            'scale': quantized_scale,
                            'scale_meta': quantized_scale_meta,
                        }
                    else:
                        quanInfo = {
                            'w': quantized_w,
                        }
                    loss = self.loss_fn(quantized_w, tmpWeight)
                    quanInfo['loss'] = loss
                    
                    now = time.time()
                    print(f"loss calculation done at {now - startTime:.2f}s")
                    print(f"{file} loss: {loss}")

                with open(pklWeightPath, 'wb') as f:
                    pickle.dump(quanInfo, f, protocol=pickle.HIGHEST_PROTOCOL)
            
                now = time.time()
                print(f"weight pickle saved at {now - startTime:.2f}s")
                print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

                # if count >=5:
                #     break

                # count += 1

        # construct the quantized model using the weight saved in the quantized dir
        if self.chunk == 9:
            rc_init_start = time.time()
            rc_model = load_model(self.model_name)            

            # if "gemma-3" in self.model_name.lower():
            rc_layers = rc_model.model.layers

            rc_linear_childern = {}

            for i in range(len(rc_layers)):
                layer = rc_layers[i]
                children = find_layers(layer)
                for weight in children.keys():
                    qtzPickleFile = self.savePath_mp_tmp_qtz / f'layer_{i}_{weight}.pickle'
                    rc_linear_childern[qtzPickleFile] = children[weight]

            print(f"rc model init done at {time.time() - rc_init_start} s")

            print(rc_linear_childern)

            # max_cpu per job fixed by slurm
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit tasks to the pool and store future objects
                future_to_rc = {executor.submit(self.reconstruct, rc_path, rc_linear_childern): rc_path for rc_path in rc_linear_childern.keys()}
                
                # Process completed tasks as they finish
                for future in concurrent.futures.as_completed(future_to_rc):
                    rc_path = future_to_rc[future]
                    nowTime = time.time()
                    try:
                        # quan_info = future.result()
                        print(f"weight {rc_path} reconstruct at {nowTime - startTime:.2f}")
                    except Exception as e:
                        print(f"weight {rc_path} reconstruction generated an exception: {str(e)}")

            # save the quantized model as pt file            
            rc_model = rc_model.to(torch.bfloat16)
            rc_model.save_pretrained(self.savePath_quanmodel)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            tokenizer.save_pretrained(self.savePath_quanmodel)
            print(f"Model saved at {time.time() - startTime} s to {self.savePath_quanmodel}")

            # delete the temp files in the original and quantized dir 
            for root, _, files in os.walk(self.savePath_mp_tmp_ori, topdown=False):
                for file in files:
                    rmPath = Path(root) / file
                    os.remove(rmPath)
            for root, _, files in os.walk(self.savePath_mp_tmp_qtz, topdown=False):
                for file in files:
                    rmPath = Path(root) / file
                    os.remove(rmPath)
            # end
            print(f"Model saved at {time.time() - startTime} s")

    @torch.no_grad()
    def run_multi_threads(self, ):
        """
        *This parallel attempt failed, it is slower than serial code for any number of workers*

        Same as run below, but run in multithreads.

        For simplicity of implamentation, each layer is a threads and save to pickle
        then the series window_grouping will read pickles and write to model parameters
        Finally save the quantized model.
        """
        startTime = time.time()
        # max_cpu = os.cpu_count()

        # may be the slow is due to lock on the same mode object
        # max_cpu per job fixed by slurm
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit tasks to the pool and store future objects
            future_to_task = {executor.submit(self.quantize_layer, layer): layer for layer in self.linear_childern.keys()}
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_task):
                layer = future_to_task[future]
                nowTime = time.time()
                try:
                    quan_info = future.result()
                    print(f"layer {layer} completed at {nowTime - startTime:.2f}: \n{quan_info}")
                except Exception as e:
                    print(f"layer {layer} generated an exception: {str(e)}")

        # save quantization result as model
        print(f"Quantization done at {time.time() - startTime} s")
        self.model.save_pretrained(self.savePath_quanmodel)
        print(f"Model saved at {time.time() - startTime} s")
        print(f"Saved to {self.savePath_quanmodel}")

    @torch.no_grad()
    def run(self, ):
        """
        execute all 
        """
        startTime = time.time()
        # save original linear layer
        # self.save_layer()

        superWeight_mask = {}
        superWeight_value = {}
        if self.model_name in self.superWeight.keys():
            superWeight_mask = self.superWeight[self.model_name]
            for layer in superWeight_mask.keys():
                superWeight_value[layer] = {}
                for linear in superWeight_mask[layer]:
                    superW_coor = superWeight_mask[layer][linear]
                    superWeight_value[layer][linear] = self.linear_childern[layer][linear].weight[superW_coor].item()

        print(superWeight_mask)
        print(superWeight_value)

        # quantization
        for layer in self.linear_childern.keys():
            # print(f"layers {layer}")
            # print(self.linear_childern[layer])

            # Save the weight of each layer as simu, binary and scale for CUDA kernal and custom layer
            quantized_linear_childern_meta = {}

            
            # Save the weight of each layer as simu, binary and scale for CUDA kernal and custom layer
            tmp_path = self.savePath_meta / f'layer_{layer}_linear.pickle'
            if not os.path.exists(tmp_path) or self.overwrite_quan:
            # if True:

                # add memory usage check for slurm mem allocation
                for linear in self.linear_childern[layer].keys():
                    print(f"layer {layer}, {linear}")

                    # print(f"layer {layer}, {linear} weight: \n{self.linear_childern[layer][linear].weight}")
                    quantized_w, quantized_bi, quantized_scale, quantized_scale_meta = self.quan(
                        self.linear_childern[layer][linear].weight
                    )

                    
                    # Save the weight of each layer as simu, binary and scale for CUDA kernal and custom layer
                    quantized_linear_childern_meta[linear] = {
                        'bi': quantized_bi,
                        'scale': quantized_scale,
                        'scale_meta': quantized_scale_meta,
                    }

                    # print(quantized_w)
                    loss = self.loss_fn(quantized_w, self.linear_childern[layer][linear].weight)
                    print('loss: ', loss)
                    # assign to the model weight
                    # testing how to realy assign to the weight
                    self.linear_childern[layer][linear].weight = nn.Parameter(quantized_w)
                    # print('weight in childern: \n', self.linear_childern[layer][linear].weight)
                    # print('weight in self.layer: \n', self.layers[0].self_attn.q_proj.weight)

                    # test quantization success
                    # tmp_class_var, inner_tmp_class_var = linear.split('.')
                    # print('weight in self.model equal to quantized weight: \n', 
                    #     torch.all(getattr(
                    #         getattr(self.model.model.layers[layer], tmp_class_var), inner_tmp_class_var
                    #     ).weight == quantized_w))
                    print(f"layer {layer}, {linear} quantized at {time.time() - startTime} s")
                    # break

                # print(f"Saving layer {layer} quantization meta data: \n", quantized_linear_childern_meta)
                print(f"Saving layer {layer} quantization meta data")
                with open(tmp_path, 'wb') as fs:
                    pickle.dump(quantized_linear_childern_meta, fs, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(tmp_path, 'rb') as f:
                    quantized_linear_childern_meta = pickle.load(f)
                    for linear in self.linear_childern[layer].keys():
                        quantized_w = quantized_linear_childern_meta[linear]['bi'] * quantized_linear_childern_meta[linear]['scale']
                        self.linear_childern[layer][linear].weight = nn.Parameter(quantized_w)
                print(f"{tmp_path} exists/skip flag (-oq) on, skipped")
            # print(f"Layer {layer} finised at {startTime - time.time()}")
            # mem usage in MiB
            print("="*50)
            print(f"layers {layer}")
            print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")
            print("="*50)
            # break

        print(superWeight_mask)
        print(superWeight_value)

        for layer in superWeight_value.keys():
            for linear in superWeight_value[layer]:
                print('Value before: ', self.linear_childern[layer][linear].weight[superW_coor])
                self.linear_childern[layer][linear].weight[superW_coor] = superWeight_value[layer][linear]
                print('Value after: ', self.linear_childern[layer][linear].weight[superW_coor])

        # save quantization result as model
        print(f"Saving model at {time.time() - startTime} s")
        print(f"Saved to {self.savePath_quanmodel}")
        self.model = self.model.to(torch.bfloat16)
        self.model.save_pretrained(self.savePath_quanmodel)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        tokenizer.save_pretrained(self.savePath_quanmodel)

    def run_v2(self, ):
        """
        Same as run but used quantize_layer to ensure quantize_layer is working,
        i.e. more OOP
        """
        startTime = time.time()
        # save original linear layer
        # self.save_layer()
        # using quantize_layer is slow
        for layer in self.linear_childern.keys():
            self.quantize_layer(layer)

        # save quantization result as model
        print(f"Quantization done at {time.time() - startTime} s")
        print(f"Saved to {self.savePath_quanmodel}")
        self.model.save_pretrained(self.savePath_quanmodel)
        print(f"Model saved at {time.time() - startTime} s")

    def evaluation(self, ):
        self.model.to(self.device)
        for dataset in ["wikitext2", "ptb", "c4"]:
            try:
                _, testloader = get_loaders(
                    dataset, seed=self.seed, seqlen=2048, model=self.model_name
                )
                print(dataset)

                eval.ppl(self.model, testloader, self.device, )
            except Exception as e:
                print(traceback.format_exc())
                pass
