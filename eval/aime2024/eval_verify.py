#!/usr/bin/env python
"""
使用math_verify重新验证MATH评估结果
"""

import json
import os
from math_verify import parse, verify
from typing import Dict, Any, List, Tuple
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def verify_math_answer(gold_answer: str, predicted_answer: str) -> bool:
    """
    使用math_verify验证数学答案是否正确
    
    Args:
        gold_answer: 标准答案
        predicted_answer: 预测答案
    
    Returns:
        bool: 是否正确
    """
    try:
        gold_answer = str(gold_answer)
        # 解析标准答案和预测答案
        gold = parse(gold_answer)
        answer = parse(predicted_answer)
        
        # 验证答案
        return verify(gold, answer)
    except Exception as e:
        # 如果解析失败，回退到字符串比较
        return gold_answer.strip() == predicted_answer.strip()

def verify_sample(sample_data: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    验证单个样本的所有回答
    
    Args:
        sample_data: 包含ground_truth和responses的样本数据
    
    Returns:
        Tuple[int, int, int]: (总回答数, 正确回答数, 是否有正确答案)
    """
    ground_truth = sample_data['ground_truth']
    responses = sample_data['responses']
    
    sample_correct_count = 0
    total_sample_responses = 0
    
    for response_data in responses:
        extracted_answer = response_data.get('extracted_answer')
        
        # 使用math_verify重新验证
        if extracted_answer is not None:
            is_correct = verify_math_answer(ground_truth, extracted_answer)
        else:
            is_correct = False
        
        if is_correct:
            sample_correct_count += 1
        
        total_sample_responses += 1
    
    has_correct = 1 if sample_correct_count > 0 else 0
    
    return total_sample_responses, sample_correct_count, has_correct

def re_evaluate_results(input_file: str, num_threads: int = 4) -> Dict[str, Any]:
    """
    重新评估MATH结果文件
    
    Args:
        input_file: 输入的结果文件路径
        num_threads: 线程数
    
    Returns:
        Dict: 评估统计信息
    """
    print(f"正在重新评估文件: {input_file}")
    print(f"使用线程数: {num_threads}")
    
    # 读取所有样本数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
    
    total_samples = len(samples)
    total_responses = 0
    correct_responses = 0
    samples_with_correct = 0
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(verify_sample, sample): sample for sample in samples}
        
        # 使用tqdm显示进度
        with tqdm(total=total_samples, desc="验证样本", unit="样本") as pbar:
            for future in as_completed(future_to_sample):
                try:
                    sample_responses, sample_correct, has_correct = future.result()
                    total_responses += sample_responses
                    correct_responses += sample_correct
                    samples_with_correct += has_correct
                except Exception as e:
                    print(f"处理样本时出错: {e}")
                finally:
                    pbar.update(1)
    
    # 计算统计信息
    overall_accuracy = correct_responses / total_responses if total_responses > 0 else 0.0
    pass_at_k_accuracy = samples_with_correct / total_samples if total_samples > 0 else 0.0
    
    stats = {
        'total_samples': total_samples,
        'total_responses': total_responses,
        'correct_responses': correct_responses,
        'samples_with_correct': samples_with_correct,
        'overall_accuracy': overall_accuracy,
        'pass_at_k_accuracy': pass_at_k_accuracy
    }
    
    print(f"重新评估完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  总回答数: {total_responses}")
    print(f"  正确回答数: {correct_responses}")
    print(f"  有正确答案的样本数: {samples_with_correct}")
    print(f"  整体准确率: {overall_accuracy:.4f}")
    print(f"  Pass@K准确率: {pass_at_k_accuracy:.4f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="使用math_verify重新验证MATH评估结果")
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入的结果文件路径')
    parser.add_argument('--summary_file', type=str,
                       help='输出的统计摘要文件路径 (默认在输入文件同目录下)')
    parser.add_argument('--num_threads', type=int, default=64,
                       help='并行验证的线程数 (默认: 64)')
    
    args = parser.parse_args()
    
    # 设置默认摘要文件路径
    if args.summary_file is None:
        base_dir = os.path.dirname(args.input_file)
        args.summary_file = os.path.join(base_dir, "verification_summary.json")
    
    # 重新评估结果
    stats = re_evaluate_results(args.input_file, args.num_threads)
    
    # 保存统计摘要
    with open(args.summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计摘要已保存到: {args.summary_file}")

if __name__ == "__main__":
    main()