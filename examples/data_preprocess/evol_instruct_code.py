"""
预处理 Evol-Instruct-Code-80k-v1 数据集为 parquet 格式
"""

import os
import datasets
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/evol_instruct_code')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=0)

    args = parser.parse_args()

    data_source = 'nickrosh/Evol-Instruct-Code-80k-v1'
    print(f"从 huggingface 加载 {data_source} 数据集...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

    # 为每个数据项添加一行，表示唯一 id
    def make_map_fn(split):
        def process_fn(example, idx):
            # 从原始数据中提取字段
            question_raw = example.get('instruction', '')
            answer_raw = example.get('output', '')
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw
                    }
                ],
                "ability": "coding",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": answer_raw,
                },
            }
            return data

        return process_fn
    

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print(f"训练数据集长度: {len(train_dataset)}")
    local_dir = args.local_dir
    if args.train_end > 0:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.train_end}.parquet'))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print(train_dataset[0])
