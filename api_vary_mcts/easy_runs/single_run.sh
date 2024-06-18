timestamp=$( date +"%Y%m%d_%H%M%S")

echo $timestamp

mkdir -p ./logs

temperature=0.5
frequency_penalty=1.2
max_iter=12

CUDA_VISIBLE_DEVICES="1" bash base_sample.sh toolqa_easy flights-easy 1 $temperature $frequency_penalty $max_iter > ./logs/flights-easy_$timestamp.log 2>&1 &

wait