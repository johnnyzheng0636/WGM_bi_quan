# Binary Quantization For LLMs Through Dynamic Grouping

## set up

Run
```
source setup.sh
```

## Run

Run with either of following commands at root. Output are default to stdout, redirect to file if necessary.

Use -h flag for more information

### Serial program

```
python ./main.py --model meta-llama/Llama-3.2-1B --quanMethod windowed_greedy --max_groups 32 --window 64 -oq
```

### Parallel with slurm
Set up your slurm config in ./parallel_main.py sbatch_template first, then
```
python parallel_main.py --model meta-llama/Llama-3.2-1B --quanMethod windowed_greedy --max_groups 32 --window 64 -oq
```

### Evaluation

#### Perplexity

Replace model_name with the local path or hugging face model

```
python ./ppl_eval.py model_name --device cpu
```
#### QA

We used [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B,trust_remote_code=True \
    --tasks piqa,boolq,openbookqa,winogrande,arc_easy,arc_challenge,hellaswag \
    --device cuda:0 \squeue
    --log_samples \
    --output_path Llama-3_2-1B \
    --batch_size auto \
    --trust_remote_code
```
#### Single matrix

Where -m is the max size of small squred matrix experiments and 
square of it is the max size of big matrix experiments

```
python matrix_test.py -m 10
```