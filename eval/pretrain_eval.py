import os, sys
os.chdir(sys.path[0])
import os.path as osp
import json
import numpy as np
import argparse


from tqdm import tqdm


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data



def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--path', type=str, default='/mnt/workspace/nas/chenguoxin.cgx/api/workspace/output_dir/greedy/run/toolqa_hard/model_cache/Qwen2-72B-Instruct')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    result = {}

    
    for file in tqdm(os.listdir(args.path)):
        if file.endswith('.jsonl') and 'final_result' not in file:
            tag = file.split('.')[0]
            file_path = osp.join(args.path, file)
            data = load_jsonl(file_path)
            acc = 0
            for item in data:
                for k, v in item.items():
                    if 'inference_epoch' in k:
                        continue
                    if len(v['solutions_tag']) != 0 and v['tree'][v['solutions_tag'][0]]['state']['reward'] == 1.0:
                        acc += 1
            result[tag] = {}
            result[tag]['acc'] = acc / len(data)
            result[tag]['acc_num'] = acc
            result[tag]['total_num'] = len(data)
    


    avg_acc = []
    for tag in result.keys():
        avg_acc.append(result[tag]['acc'])
    result['avg_acc'] = np.mean(avg_acc)

    with open(osp.join(args.path, f'{osp.basename(args.path)}.json'), 'w') as f:
        json.dump(result, f, indent=4)

                    
            
    