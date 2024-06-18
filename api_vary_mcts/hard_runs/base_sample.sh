task=$1
dataname=$2
num_epoch=$3
echo $task
echo $dataname
echo $num_epoch
temperature=$4
frequency_penalty=$5
max_iter=$6

python=/opt/conda/envs/vary/bin/python
checkpoint_dir=/mnt/workspace/nas/chenguoxin.cgx/model_cache/Meta-Llama-3-8B-Instruct

export VLLM_USE_MODELSCOPE="False"

debug_num=-1
filter=False
filter_path="11"
output_dir=/mnt/workspace/nas/chenguoxin.cgx/api/workspace/output_dir/mcts/round1/run

seed=0

$python ../src/batch_search_generate.py \
    --debug_num $debug_num \
    --task $task \
    --dataname $dataname \
    --output_dir $output_dir \
    --seed $seed \
    --num_epoch $num_epoch \
    --filter $filter \
    --filter_path $filter_path \
    --checkpoint_dir $checkpoint_dir \
    --temperature $temperature \
    --frequency_penalty $frequency_penalty \
    --max_iter $max_iter

