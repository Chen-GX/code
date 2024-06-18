

mkdir -p ./logs

CUDA_VISIBLE_DEVICES="1" bash sample.sh toolqa_hard agenda-hard 2 > ./logs/agenda-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="2" bash sample.sh toolqa_hard airbnb-hard 2 > ./logs/airbnb-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="3" bash sample.sh toolqa_hard coffee-hard 2 > ./logs/coffee-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="4" bash sample.sh toolqa_hard dblp-hard 2 > ./logs/dblp-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="5" bash sample.sh toolqa_hard flights-hard 2 > ./logs/flights-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="6" bash sample.sh toolqa_hard scirex-hard 2 > ./logs/scirex-hard.log 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES="7" bash sample.sh toolqa_hard yelp-hard 2 > ./logs/yelp-hard.log 2>&1 &
wait