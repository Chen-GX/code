timestamp=$(date +"%Y%m%d_%H%M%S")
echo $timestamp

mkdir -p ./logs

ckpt_list=(
    /mnt/workspace/nas/chenguoxin.cgx/model_cache/Meta-Llama-3-8B-Instruct
    /mnt/workspace/nas/chenguoxin.cgx/model_cache/Meta-Llama-3-8B
)

task=toolqa_easy

dataname_lists="agenda-easy,airbnb-easy,coffee-easy,dblp-easy,flights-easy,scirex-easy,yelp-easy"

sft_prompt=True

# ckpt的数量
num_ckpt=${#ckpt_list[@]}
for ((i=0;i<$num_ckpt;i++))
do
    # 取出第i个元素
    ckpt=${ckpt_list[$i]}
    ckpt_dirname=$(basename "$(dirname "$ckpt")")
    ckpt_basename=$(basename "$ckpt")
    CUDA_VISIBLE_DEVICES=1 bash base_sample.sh $task $dataname_lists "$ckpt" $sft_prompt > "./logs/${ckpt_dirname}_${ckpt_basename}_${timestamp}.log" 2>&1
    wait
    sleep 60
done
wait
