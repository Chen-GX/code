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
    parser.add_argument('--path', type=str, default='/mnt/workspace/nas/chenguoxin.cgx/api/workspace/output_dir/greedy/run/toolqa_easy/20240620_001056')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    result = {}

    for ckpt_name in tqdm(os.listdir(args.path)):
        if ckpt_name.startswith('checkpoint'):
            if ckpt_name not in result.keys():
                result[ckpt_name] = {}

            ckpt_path = osp.join(args.path, ckpt_name)
            # 列出所有的文件
            for file in os.listdir(ckpt_path):
                if file.endswith('.jsonl') and 'final_result' not in file:
                    tag = file.split('.')[0]
                    file_path = osp.join(ckpt_path, file)
                    data = load_jsonl(file_path)
                    acc = 0
                    for item in data:
                        for k, v in item.items():
                            if 'inference_epoch' in k:
                                continue
                            if len(v['solutions_tag']) != 0 and v['tree'][v['solutions_tag'][0]]['state']['reward'] == 1.0:
                                acc += 1
                    result[ckpt_name][tag] = {}
                    result[ckpt_name][tag]['acc'] = acc / len(data)
                    result[ckpt_name][tag]['acc_num'] = acc
                    result[ckpt_name][tag]['total_num'] = len(data)
    

    for ckpt in result.keys():
        avg_acc = []
        for tag in result[ckpt].keys():
            avg_acc.append(result[ckpt][tag]['acc'])
        result[ckpt]['avg_acc'] = np.mean(avg_acc)

    with open(osp.join(args.path, 'final_result.json'), 'w') as f:
        json.dump(result, f, indent=4)

                    
            
    