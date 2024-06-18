# 收集各个数据集中，已经采样出正确solution的问题数目，并且把相应的问题的id进行记录，在下一轮采样中跳过这些问题

import os, sys
os.chdir(sys.path[0])
import os.path as osp
import json
import glob

num_round = './round1'

os.makedirs(num_round, exist_ok=True)

def load_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

check_path = '/mnt/workspace/nas/chenguoxin.cgx/api/workspace/output_dir/mcts/round1/run/toolqa_easy'
modelname = 'Meta-Llama-3-8B-Instruct'  # 用于采样的模型
threshold = 4

count_result = {}
datanames = os.listdir(check_path)

for dataname in datanames:
    count_result[dataname] = {"num": 0, "question_ids": []}  # 每个数据集一共有多少问题已经有正确的solution了，
    for timestampe in os.listdir(osp.join(check_path, dataname, modelname)):
        file_dir = osp.join(check_path, dataname, modelname, timestampe, "collectted_solutions")
        for file_path in glob.glob(f'{file_dir}/collectted_solutions_*.jsonl'):
            data = load_jsonl(file_path)
            print(file_dir)
            for item in data:
                # 知道是哪个id
                for k, v in item.items():
                    if 'inference' not in k:
                        question_id = k
                        break
                
                num = 0
                solution = item[question_id]
                for tag in solution['solutions_tag']:
                    if solution['tree'][tag]['state']['reward'] == 1.0:
                        num += 1
                if num >= threshold:
                    if question_id not in count_result[dataname]["question_ids"]:
                        count_result[dataname]["question_ids"].append(question_id)
    
    # 更新num数
    count_result[dataname]['num'] = len(count_result[dataname]["question_ids"])


# 存储文件
with open(osp.join(f"{num_round}", f"{osp.basename(check_path)}_{threshold}.json"), 'w') as f:
    json.dump(count_result, f, indent=2)

num_count = {}
for dname, item in count_result.items():        
    num_count[dname] = {}  
    for k, v in item.items():
        if k == "num":
            num_count[dname][k] = v



with open(osp.join(f"{num_round}", f"{osp.basename(check_path)}_{threshold}_num.json"), 'w') as f:
    json.dump(num_count, f, indent=2)
