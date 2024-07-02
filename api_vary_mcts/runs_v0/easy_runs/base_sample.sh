task=$1
dataname=$2
num_epoch=$3
echo $task
echo $dataname
echo $num_epoch
temperature=$4
frequency_penalty=$5
max_iter=$6

root_path=/mnt/workspace/nas/chenguoxin.cgx
# root_path=/yinxr/workhome/zzhong/chenguoxin
# root_path=/bjzhyai03/workhome/cgx/chenguoxin


python=/opt/conda/envs/vary/bin/python
# python=/yinxr/workhome/zzhong/miniconda3/envs/vary/bin/python
# python=/bjzhyai03/workhome/cgx/miniconda3/envs/vary/bin/python

checkpoint_dir=${root_path}/model_cache/Meta-Llama-3-8B-Instruct

export VLLM_USE_MODELSCOPE="False"

debug_num=-1
filter=False
filter_path=${root_path}/api/workspace/code/api_vary_mcts/count_num/easy/toolqa_easy_4.json
output_dir=${root_path}/api/workspace/output_dir/mcts/round1_v0/run

api_kernel_version=0

seed=0

$python ../../src/batch_search_generate.py \
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
    --max_iter $max_iter \
    --api_kernel_version $api_kernel_version

