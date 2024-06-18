timestamp=$( date +"%Y%m%d_%H%M%S")

echo $timestamp

mkdir -p ./logs

temperature=0.5
frequency_penalty=1.2
max_iter=12

CUDA_VISIBLE_DEVICES="1" bash base_sample.sh toolqa_easy dblp-easy 1 $temperature $frequency_penalty $max_iter> ./logs/dblp-easy_$timestamp.log 2>&1 &
sleep 10

CUDA_VISIBLE_DEVICES="2" bash base_sample.sh toolqa_easy dblp-easy 1 $temperature $frequency_penalty $max_iter> ./logs/dblp-easy_2_$timestamp.log 2>&1 &
sleep 10

CUDA_VISIBLE_DEVICES="3" bash base_sample.sh toolqa_easy dblp-easy 1 $temperature $frequency_penalty $max_iter> ./logs/dblp-easy_3_$timestamp.log 2>&1 &
sleep 10

wait