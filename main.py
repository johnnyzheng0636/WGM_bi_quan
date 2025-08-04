#  call quna to compress LLMs
import psutil
import torch
from torch import nn
import argparse
from my_util import load, eval
import time
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B", help="model to load; for example `meta-llama/Llama-3.2-1B`."
    )
    parser.add_argument(
        "--quanMethod", type=str, default="windowed_greedy", help="Algorithm to use, only Agitlgo3 is available, Algo1: Algo2: Algo3: windowed_greedy"
    )
    
    parser.add_argument(
        "--layer_cache_dir", type=str, default="./hidden_data", help="directory to save the layer cache for experiments."
    )
    # tmp storage
    # parser.add_argument(
    #     "--layer_cache_dir", type=str, default="/project/mscbdt2024/xzhengbj/5014/bi_quan/hidden_data", help="directory to save the layer cache for experiments."
    # )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling perplexity data."
    )
    parser.add_argument(
        "--window", type=int, default=512, help="Window size for windowed greedy algorithm."
    )
    parser.add_argument(
        "--max_groups", type=int, default=32, help="Max groups for dynamic grouping."
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=0.75, help="lambda for regularization and auto group size finding"
    )
    parser.add_argument(
        "-ab", "--avg_bit", type=int, default=None, help="Overwrite --max_groups"
    )


    parser.add_argument(
        "-oo", "--overwrite_origin", action='store_true', help="overwrite original layers"
    )
    parser.add_argument(
        "-oq", "--overwrite_quan", action='store_true', help="overwrite quantized layers"
    )
    parser.add_argument(
        "-eq", "--use_quantized_model", action='store_true', help="only load quantized modoel or origin model for evaluation"
    )
    # not tested not device with enough memories and cpus
    parser.add_argument(
        "-mt", "--multi_thread", action='store_true', help="use multi thread to run"
    )
    parser.add_argument(
        "-ck", "--chunk",type=int, default=0, help="Only for -mt. Which chunk to quantize for multi thread, 1-8, 0 = init, 9 = reconstruct"
    )
    parser.add_argument(
        "-ep", "--eval_ppl", action='store_true', help="evaluate quantization perplexity"
    )
    # parser.add_argument(
    #     "-qa", "--qa_eval_only", action='store_true', help="only evaluate on QA task"
    # )

    print('cpu counts: ', os.cpu_count())

    args = parser.parse_args()

    print('='*50)
    print(f"using {args.model}")
    print('='*50)

    # load.save_layer(args.model, args.layer_cache_dir)
    start = time.time()


    # if not args.qa_eval_only:
    if args.multi_thread:
        outpath = args.layer_cache_dir + '_mt'
    else:
        outpath = args.layer_cache_dir

    print('mem at the start')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

    # load model
    bi_quan = load.binaryQuantization(
        args.model, 
        outpath, 
        args.quanMethod, 
        lambda_reg=args.lambda_reg,
        overwrite_quan=args.overwrite_quan,
        seed=args.seed,
        evalonly=args.use_quantized_model,
        max_groups=args.max_groups,
        window=args.window,
        chunk=args.chunk,
        avgBit=args.avg_bit,
    )

    print('mem at after init')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, "GiB")

    # quantize model
    if not args.use_quantized_model:
        if args.multi_thread:
            print("parallel quantization")
            # deprecated
            # bi_quan.run_multi_threads()
            bi_quan.run_mp_per_weight()
        else:
            print("serial quantization")
            bi_quan.run()

    # evaluation model perplexity
    if args.eval_ppl:
        bi_quan.evaluation()

    # eval.eval_qa(args.model)

    end = time.time()
    print('Qunatization used: ', end - start, 's')
