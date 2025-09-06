import subprocess
import argparse
from my_util import load, eval
import time
import os
from pathlib import Path
import sys
import shlex
# python multi threading is slow, so parallel is done by submiting multiple slurm jobs
# This .py create the corresponding .sbatch for each jobs, and run them in the
# correct sequence.

# restor the input args
def restore_args(args):
    pass

# sbatch file content
def sbatch_template(out_dir, out_file, py_args_str):
    # replace #SBATCH with the your sbatch content
    return fr"""#!/bin/bash
#SBATCH --job-name=biQ_C_mp      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=cpu          # partition(large/normal/cpu) where you submit
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=1        # number of max cpu per job
#SBATCH --mem-per-cpu=64G                # memory per node, alternatively can use --mem-per-cpu for mem of each cpu
#SBATCH --account=your_account   # only require for multiple projects
#SBATCH --output=./stdout/cpu_parallel/{out_dir}/{out_file}.out       #output file

echo "start"
date

# TODO paralell test
# tested for 4 bit on average 5 is the largerst init window size for llama
python ./main.py {py_args_str}

date
echo "end"
"""

# take the same input as main.py, but must have -mt flag on
if __name__ == '__main__':
    start = time.time()

    raw_args_list = sys.argv[1:]
    raw_args_str = ' '.join(shlex.quote(arg) for arg in raw_args_list) + ' --multi_thread -ck '


    # argsparse ensure correct input args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B", help="model to load; for example `meta-llama/Llama-3.2-1B`."
    )
    parser.add_argument(
        "--quanMethod", type=str, default="windowed_greedy", help="model to load; for example `meta-llama/Llama-3.2-1B`."
    )
    
    parser.add_argument(
        "--layer_cache_dir", type=str, default="./hidden_data", help="directory to save the layer cache for experiments."
    )

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
    # must be ture for parallel_main
    # parser.add_argument(
    #     "-mt", "--multi_thread", action='store_true', help="use multi thread to run"
    # )
    # automatically assigne 0 to 9
    # parser.add_argument(
    #     "-ck", "--chunk",type=int, default=0, help="Only for -mt. Which chunk to quantize for multi thread, 1-8, 0 = init, 9 = reconstruct"
    # )
    parser.add_argument(
        "-ep", "--eval_ppl", action='store_true', help="evaluate quantization perplexity"
    )
    # parser.add_argument(
    #     "-qa", "--qa_eval_only", action='store_true', help="only evaluate on QA task"
    # )

    print('cpu counts: ', os.cpu_count())

    args = parser.parse_args()

    print(args)
    print(raw_args_str)

    # create the dirctory containing all sbatch files
    dir_name = f"{args.model.split('/')[-1]}"
    if args.avg_bit is not None:
        dir_name += f"_{args.avg_bit}b"
    else:
        dir_name += f"_{args.max_groups}g"
    dir_name += f"_{args.lambda_reg}l"
    dir_name += f"_{args.window}w"
    job_dir = Path(f"./job/{dir_name}")
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"job_dir: {job_dir}")

    # create all sbatch files
    for chunk in range(10):
        file_name = f"mt_{chunk}.sbatch"
        dst_path = job_dir / file_name
        chunk_args_string = raw_args_str + str(chunk)
        with open(dst_path, 'w') as f:
            f.write(sbatch_template(dir_name, chunk, chunk_args_string))

    # run the initial sbatch file and wait for init to complete
    print(f'init start at {time.time() - start} seconds')
    init_sbatch_path = job_dir / "mt_0.sbatch"
    job_init_id = subprocess.run(f'sbatch --wait {init_sbatch_path}', shell=True, capture_output=True)
    job_init_id = job_init_id.stdout.decode().split(' ')[-1].strip()
    print(f'init job id: {job_init_id}')
    print(f'init done at {time.time() - start} seconds')

    # run the worker sbatch files
    job_id_ls = []
    for chunk in range(1,9):
        sbatch_path = job_dir / f"mt_{chunk}.sbatch"
        job_tmp_id = subprocess.run(f'sbatch {sbatch_path}', shell=True, capture_output=True)
        job_id_ls.append(job_tmp_id.stdout.decode().split(' ')[-1].strip())
    job_id_str = ','.join(job_id_ls)

    # wait for all workers to complete
    while 1:
        squeue_info = subprocess.run(f'squeue -j {job_id_str} -h', shell=True, capture_output=True)
        squeue_info = squeue_info.stdout.decode()
        if squeue_info == '':
            break
        time.sleep(10)
    
    print(f'All workers finished at {time.time() - start} seconds')
    
    # run the join sbatch file
    join_sbatch_path = job_dir / "mt_9.sbatch"
    subprocess.run(f'sbatch --wait {join_sbatch_path}', shell=True)
    print(f'Join finished at {time.time() - start} seconds')

    # end
    print(f'Stdout of all jobs are saved to {job_dir}')
