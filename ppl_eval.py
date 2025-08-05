# evaluates perplexityu of a local model given the path to the model
import time

import torch
import torch.nn as nn

from pprint import pprint
import traceback

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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `meta-llama/Llama-3.2-1B`."
    )
    parser.add_argument(
        "--load_cache_model", 
        type=str, 
        default='',
        help="model cache path."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization. cpu or cuda:x",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )


    args = parser.parse_args()

    print('after parser')

    device = args.device
    print("loading model ", args.model)
    model = get_model(args.model)
    print("model loaded")
    model.eval()

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