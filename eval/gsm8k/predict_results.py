#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json
import os

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_answer(solution_str):
    # 只匹配 \boxed{...} 中的内容
    match = re.search(r'\\boxed\{([^\}]+)\}', solution_str)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def evaluate_gsm8k_answer(generated_answer, ground_truth):
    """评估GSM8K答案是否正确"""
    if generated_answer is None:
        return False
    
    try:
        # 将答案转换为浮点数进行比较
        gen_num = float(generated_answer.replace(",", ""))
        gt_num = float(ground_truth.replace(",", ""))
        return abs(gen_num - gt_num) < 1e-6
    except:
        # 如果转换失败，进行字符串比较
        return generated_answer.strip() == ground_truth.strip()


@dataclass
class ScriptArguments:
    """
    GSM8K评估脚本的参数
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "GSM8K数据集路径"},
    )
    dataset_split: Optional[str] = field(
        default="test",
        metadata={"help": "数据集分割（train或test）"},
    )
    output_dir: Optional[str] = field(
        default="./gsm8k_eval_results/",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=1,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=1,
        metadata={"help": "每个问题生成的回答数量"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.0,  # 为了确定性结果，使用0温度
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("模型路径:", model_path)
seed = script_args.seed
# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 加载模型
llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id],
)

# 加载GSM8K数据集
print(f"加载GSM8K数据集: {script_args.dataset_name_or_path}")
ds = load_dataset(script_args.dataset_name_or_path, "main", split=script_args.dataset_split)

# 添加说明文本
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

# 处理数据集
def process_gsm8k_data(example):
    question = example["question"] + " " + instruction_following
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 提取ground truth答案
    ground_truth = extract_solution(example["answer"])
    
    return {
        "prompt": prompt,
        "question": example["question"],
        "answer": example["answer"],
        "ground_truth": ground_truth
    }

ds = ds.map(process_gsm8k_data)

# 生成回答
prompts = ds["prompt"]
print("开始生成回答...")
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

# 评估结果
evaluation_results = []
correct_count = 0
total_count = len(outputs)

for i, output in enumerate(outputs):
    ground_truth = ds[i]["ground_truth"]
    responses = [out.text for out in output.outputs]
    
    # 对每个生成的回答进行评估
    response_evaluations = []
    for response in responses:
        generated_answer = extract_answer(response)
        is_correct = evaluate_gsm8k_answer(generated_answer, ground_truth)
        response_evaluations.append({
            "response": response,
            "extracted_answer": generated_answer,
            "is_correct": is_correct
        })
    
    # 检查是否有任何回答正确（对于多次生成的情况）
    any_correct = any(eval_result["is_correct"] for eval_result in response_evaluations)
    if any_correct:
        correct_count += 1
    
    result = {
        "question": ds[i]["question"],
        "ground_truth": ground_truth,
        "responses": response_evaluations,
        "any_correct": any_correct
    }
    evaluation_results.append(result)

# 计算准确率
accuracy = correct_count / total_count if total_count > 0 else 0

os.makedirs(script_args.output_dir, exist_ok=True)

# 保存详细结果
output_file = os.path.join(script_args.output_dir, f"gsm8k_eval_results.jsonl")
with open(output_file, "w", encoding="utf8") as f:
    for result in evaluation_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# 保存汇总结果
summary_file = os.path.join(script_args.output_dir, f"gsm8k_summary.json")
summary = {
    "model_path": model_path,
    "dataset_split": script_args.dataset_split,
    "total_count": total_count,
    "correct_count": correct_count,
    "accuracy": accuracy,
    "parameters": {
        "temperature": script_args.temperature,
        "max_new_tokens": script_args.max_new_tokens,
        "K": script_args.K
    }
}

with open(summary_file, "w", encoding="utf8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(summary)
