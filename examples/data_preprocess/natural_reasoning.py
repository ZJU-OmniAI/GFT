"""
预处理 Natural Reasoning 数据集为 parquet 格式
"""

import os
import datasets
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/natural_reasoning')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=0)

    args = parser.parse_args()

    data_source = 'facebook/natural_reasoning'
    print(f"从 huggingface 加载 {data_source} 数据集...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

    # 为每个数据项添加一行，表示唯一 id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('question')
            reference_answer = example.pop('reference_answer')
            responses = example.pop('responses')
            
            # 从 responses 中提取第一个回答作为答案
            answer = responses[0]['response'] if responses and len(responses) > 0 else ""
                
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": reference_answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "reference_answer": reference_answer,
                    "responses": responses,
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
