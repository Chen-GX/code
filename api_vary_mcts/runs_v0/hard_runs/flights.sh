timestamp=$( date +"%Y%m%d_%H%M%S")

echo $timestamp

mkdir -p ./logs

temperature=0.5
frequency_penalty=1.2
max_iter=12
# max_depth=12

cd /mnt/workspace/nas/chenguoxin.cgx/api/workspace/code/api_vary_mcts/src
bash /mnt/workspace/nas/chenguoxin.cgx/api/workspace/code/api_vary_mcts/src/start_gunicorn.sh &
sleep 600

cd /mnt/workspace/nas/chenguoxin.cgx/api/workspace/code/api_vary_mcts/hard_runs

CUDA_VISIBLE_DEVICES="1" bash base_sample.sh toolqa_hard flights-hard 1 $temperature $frequency_penalty $max_iter > ./logs/flights-hard_$timestamp.log 2>&1 &
sleep 20

wait