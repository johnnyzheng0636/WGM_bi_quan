# similar to run.py, but this only evaluates perplexityu of 
# a local model given the path to the model
import time

import torch
import torch.nn as nn

# from bigptq import BRAGPTQ
# from binary import Binarization
# from modelutils import find_layers
# from huggingface_hub import login

from pprint import pprint
import traceback

# trun it off before submit slurm job

# DEBUG = True        # before ready
# DEBUG_ar = True     # after ready

# DEBUG = False        # before ready
# DEBUG_ar = False     # after ready

# login(token = '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')

tstart = time.time()

def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model.lower():
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model.lower():
        from transformers import AutoModelForCausalLM

        # if args.debug:
        #     print('calling transformer to get model')
        if args.load_cache_model != '':
            print('loading cache explicitly')
            model = AutoModelForCausalLM.from_pretrained(args.load_cache_model, torch_dtype="auto")
        else:
            print('loading model by default')
            # below is very slow, using above directly use cache
            print(model)
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        # if args.debug:
        #     print('transformer got model')
        model.seqlen = 2048
    elif "gemma" in model.lower():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        # token length for calibration
        model.seqlen = 2048
    elif "falcon" in model.lower():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        model.seqlen = 2048
    return model

if __name__ == "__main__":
    import argparse
    from datautils import *

    # def list_of_ints(arg):
    #     return list(map(int, arg.split(',')))
    
    # def list_of_floats(arg):
    #     return list(map(float, arg.split(',')))
    # if DEBUG:
    #     print('before parser')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    # parser.add_argument(
    #     "dataset",
    #     type=str,
    #     choices=["wikitext2", "ptb", "c4"],
    #     help="Where to extract calibration data from.",
    # )
    # parser.add_argument(
    #     "low_quant_method",
    #     type=str,
    #     choices=["xnor", "sign", "no", "1bit", "2bit", "3bit", "4bit", "prune", "braq", "1.58bit"],
    #     help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    # )
    parser.add_argument(
        "--load_cache_model", 
        type=str, 
        default='',
        help="model cache path; for example '/home/xzhengbj/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'."
    )
    # parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    # parser.add_argument(
    #     "--nsamples", type=int, default=128, help="Number of calibration data samples."
    # )
    # parser.add_argument(
    #     "--percdamp",
    #     type=float,
    #     default=0.01,
    #     help="Percent of the average Hessian diagonal to use for dampening.",
    # )
    # parser.add_argument(
    #     "--blocksize",
    #     type=int,
    #     default=128,
    #     help="Blocksize to use for adaptive mask selection.",
    # )
    # parser.add_argument(
    #     "--salient_metric",
    #     type=str,
    #     default="magnitude",
    #     choices=["magnitude", "hessian"],
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization. cpu or cuda:x",
    )
    # parser.add_argument(
    #     "--disable_gptq",
    #     action="store_true",
    #     help="disable GPTQ for quantization.",
    # )
    # parser.add_argument(
    #     "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    # )
    # parser.add_argument(
    #     "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    # )
    # parser.add_argument(
    #     "--quant_only",
    #     type=str,
    #     default="",
    #     help="Quant only layers that contain this text.",
    # )
    # parser.add_argument("--invert", action="store_true", help="Invert subset.")
    # parser.add_argument(
    #     "--save",
    #     action="store_true",
    # )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    # parser.add_argument(
    #     "--catcher", action="store_false", help="test new catcher"
    # )
    # parser.add_argument(
    #     "--debug", action="store_true", help="debug mode"
    # )
    # parser.add_argument(
    #     "--o1", type=int, default=1, help="order[0]"
    # )
    # parser.add_argument(
    #     "--o2", type=int, default=1, help="order[1]"
    # )
    # parser.add_argument(
    #     "--o3", type=int, default=2, help="order[2]"
    # )

    args = parser.parse_args()
    # groupsize = args.blocksize

    # order = (args.o1, args.o2, args.o3)

    # if args.device != 'cpu':
    #     DEBUG = False
    #     DEBUG_ar = False

    # if DEBUG:
    print('after parser')

    device = args.device
    # if order != (1,1,2):
    #     save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_order{args.o1}{args.o2}{args.o3}"
    # else:        
    #     save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    # save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    print("loading model ", args.model)
    # if args.load_quantized:
    model = get_model(args.model)
    print("model loaded")
    model.eval()
    # else: # braq
    #     print("only for quantized/local model perplexity evaluation")
    #     exit(0)

    print('quantization finished/loaded with {} seconds'.format(time.time() - tstart))

    for dataset in ["wikitext2", "ptb", "c4"]:
        try:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
            )
            print(dataset)
            model_name = args.model.split("/")[-1].lower()
            if "opt" in model_name:
                from eval_ppl_utils import opt_eval

                opt_eval(model, testloader, device, dataset, args.log_wandb)
            elif "llama" in model_name:
                from eval_ppl_utils import llama_eval

                llama_eval(model, testloader, device, dataset, args.log_wandb)
            elif "falcon" in model_name:
                from eval_ppl_utils import falcon_eval

                falcon_eval(model, testloader, device, dataset, args.log_wandb)
            elif "gemma-3" in model_name:
                from eval_ppl_utils import gemma3_1b_eval

                gemma3_1b_eval(model, testloader, device, dataset, args.log_wandb)
            elif "gemma-3-4" in model_name:
                from eval_ppl_utils import gemma3_4b_eval

                gemma3_4b_eval(model, testloader, device, dataset, args.log_wandb)
        except Exception as e:
            print(traceback.format_exc())
            pass