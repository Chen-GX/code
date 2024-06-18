
mkdir -p ./logs

for task in toolqa_easy
do
    for dataname in agenda-easy airbnb-easy coffee-easy dblp-easy flights-easy scirex-easy yelp-easy
    do
        CUDA_VISIBLE_DEVICES=7 bash sample.sh $task $dataname > ./logs/$dataname.log 2>&1
        sleep 30
    done
done


#  airbnb-easy  coffee-easy flights-easy scirex-easy 