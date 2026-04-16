# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import json
import copy
import os
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )

def compute_forward_kl_divergence(log_probs_base, log_probs_model, response_mask):
    """
    计算前向 KL 散度 (Forward KL Divergence): KL(Base || Model)
    基于论文 "RL's Razor" 中的定义: E[KL(pi_0 || pi)]
    
    参数:
        log_probs_base (torch.Tensor): 基础模型(Reference Model)对轨迹token的log概率。
                                    Shape: [Batch_Size, Seq_Len] (e.g., [512, 2048])
        log_probs_model (torch.Tensor): 微调模型(Current Model)对轨迹token的log概率。
                                        Shape: [Batch_Size, Seq_Len] (e.g., [512, 2048])
        response_mask (torch.Tensor):  掩码矩阵，指示哪些token是有效的响应部分（排除Prompt和Padding）。
                                    Shape: [Batch_Size, Seq_Len] (e.g., [512, 2048])
                                    1 表示有效 token，0 表示被掩盖的 token。
    
    返回:
        mean_kl (torch.Tensor): 批次内的平均 KL 散度值 (Scalar)。
    """
    
    # 1. 计算逐 Token 的 Log 概率差值 (Forward KL: log_p_base - log_p_model)
    # 对应公式: log(pi_0(y)/pi(y)) = log(pi_0(y)) - log(pi(y))
    token_level_kl = log_probs_base - log_probs_model
    
    # 2. 应用掩码 (Response Mask)
    # 只保留回答部分(response)的差异，忽略 Prompt 和 Padding
    masked_kl = token_level_kl * response_mask
    
    # 3. 计算序列级 KL 散度 (Sum over Sequence)
    # 将一条轨迹上所有有效 Token 的 KL 值求和，得到该序列的 KL 散度
    # dim=1 代表沿序列长度维度求和
    sequence_level_kl = masked_kl.sum(dim=1)
    
    # 4. 计算批次平均值 (Expectation over Batch)
    # 对应公式中的 E_x~tau [...]
    # 这里假设 response_mask 已经处理好了 batch 中有效数据的筛选，或者 batch 是均匀采样的
    mean_kl = sequence_level_kl.mean()
    
    return mean_kl

def concat_dataproto_objects(dataproto1: DataProto, dataproto2: DataProto) -> DataProto:
    """
    合并两个 DataProto 对象。
    
    Args:
        dataproto1: 第一个 DataProto 对象（如 gen_batch_union）
        dataproto2: 第二个 DataProto 对象（如 expert_batch_union）
    
    Returns:
        合并后的 DataProto 对象
    """
    
    # 1. 合并 batch（TensorDict）中的 tensor
    merged_batch_dict = {}
    for key in dataproto1.batch.keys():
        if key in dataproto2.batch.keys():
            # 获取两个 tensor
            tensor1 = dataproto1.batch[key]
            tensor2 = dataproto2.batch[key]
            
            # 确保都是 tensor
            if not isinstance(tensor1, torch.Tensor):
                tensor1 = torch.tensor(tensor1)
            if not isinstance(tensor2, torch.Tensor):
                tensor2 = torch.tensor(tensor2)
            
            # 在 dim=0（batch 维度）进行 concat
            merged_batch_dict[key] = torch.cat([tensor1, tensor2], dim=0)
        else:
            print(f"Warning: key '{key}' not found in dataproto2.batch")
    
    # 检查 dataproto2 中是否有 dataproto1 没有的 key
    for key in dataproto2.batch.keys():
        if key not in dataproto1.batch.keys():
            print(f"Warning: key '{key}' in dataproto2.batch but not in dataproto1.batch")
    
    # 2. 合并 non_tensor_batch（dict）中的 numpy array
    merged_non_tensor_dict = {}
    for key in dataproto1.non_tensor_batch.keys():
        if key in dataproto2.non_tensor_batch.keys():
            # 获取两个 numpy array
            arr1 = dataproto1.non_tensor_batch[key]
            arr2 = dataproto2.non_tensor_batch[key]
            
            # 确保都是 numpy array
            if not isinstance(arr1, np.ndarray):
                arr1 = np.array(arr1)
            if not isinstance(arr2, np.ndarray):
                arr2 = np.array(arr2)
            
            # 在 axis=0（第一维）进行 concat
            merged_non_tensor_dict[key] = np.concatenate([arr1, arr2], axis=0)
        else:
            print(f"Warning: key '{key}' not found in dataproto2.non_tensor_batch")
    
    # 检查 dataproto2 中是否有 dataproto1 没有的 key
    for key in dataproto2.non_tensor_batch.keys():
        if key not in dataproto1.non_tensor_batch.keys():
            print(f"Warning: key '{key}' in dataproto2.non_tensor_batch but not in dataproto1.non_tensor_batch")
    
    # 3. 创建新的 TensorDict
    new_batch = TensorDict(merged_batch_dict, batch_size=len(merged_batch_dict[list(merged_batch_dict.keys())[0]]))
    
    # 4. 合并 meta_info（通常为空 dict，如果有内容则合并）
    merged_meta_info = {}
    if dataproto1.meta_info:
        merged_meta_info.update(dataproto1.meta_info)
    if dataproto2.meta_info:
        merged_meta_info.update(dataproto2.meta_info)
    
    # 5. 创建新的 DataProto 对象
    merged_dataproto = DataProto(
        batch=new_batch,
        non_tensor_batch=merged_non_tensor_dict
    )
    if merged_meta_info:
        merged_dataproto.meta_info = merged_meta_info
    
    return merged_dataproto


def _repeat_interleave(value, repeats: int):
    """兼容 torch.Tensor 或 numpy.ndarray 或 Python list 的重复展开（按第0维重复）。"""
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        # for list / numpy array: convert to list and repeat
        if hasattr(value, "__len__"):
            return [item for item in value for _ in range(repeats)]
        else:
            raise TypeError("_repeat_interleave only supports tensor/list/ndarray")

def build_dataprop_from_gen_batch(
    gen_batch: DataProto,
    response_length: int,
    pad_token_id: int,
    calculate_log_probs: bool = False,
    n_candidates: Optional[int] = None,
) -> DataProto:
    """
    将 `gen_batch`（包含 prompt tokens + extra_info 中的多候选 model_answer_id）转换为与
    vLLM rollout 相同结构的 DataProto：
      batch: TensorDict with keys "prompts","responses","input_ids","attention_mask","position_ids"
      non_tensor_batch: 与输入 gen_batch.non_tensor_batch 基本一致，但会把与样本个数相关的字段展开以匹配重复后的样本数

    行为说明：
    - 对每个原始样本 i，读取 gen_batch.non_tensor_batch["extra_info"][i]["model_answer_id"],
      这是一个候选答案列表（例如长度为 8）。函数会把这些候选逐一展开，形成新的 batch，
      等同于把 sampling n = number_of_candidates 的生成结果展开。
    - 对展开后的 responses 做右侧 padding 到 `response_length`（使用 `pad_2d_list_to_length`）。
    - 对 prompts/attention/position 做相同次数的 repeat_interleave。
    - 计算并拼接新的 `input_ids = cat(prompt, response)`，以及更新 position_ids、attention_mask（通过 get_response_mask）。
    - 如果 `calculate_log_probs` 为 True，函数会尝试从 `extra_info` 中读取对应的 logprob（如果存在），
      否则不会创建 `rollout_log_probs` 字段。

    参数：
      gen_batch: 输入 DataProto
      response_length: 目标响应长度（填充到该长度）
      pad_token_id: 用于填充 response 的 token id
      calculate_log_probs: 是否尝试构建 rollout_log_probs（通常为 False）
      n_candidates: 如果提供，则对每个样本仅使用前 n_candidates 个候选；否则使用每样本全部候选。

    返回：
      DataProto（新构造的 batch 与 non_tensor_batch）
    """
    # breakpoint()
    # 从 gen_batch 提取 tensor 部分
    idx = gen_batch.batch["input_ids"]   # (bs, prompt_len)
    attention_mask = gen_batch.batch["attention_mask"]
    position_ids = gen_batch.batch["position_ids"]
    device = idx.device
    orig_bs = idx.size(0)

    non_tensor = dict(gen_batch.non_tensor_batch)  # shallow copy
    non_tensor_batch_remove_keys = ["extra_info", "raw_prompt_ids"]
    extra_info = non_tensor.get("extra_info", None)
    if extra_info is None:
        raise ValueError("gen_batch.non_tensor_batch must contain 'extra_info' with 'model_answer_id'")

    # 构造所有响应 candidates 的列表（扁平化）
    responses_list = []            # list[list[int]] 每个元素是一个候选答案（未填充）
    rollout_log_probs_list = []    # 如果存在相应 logprob 信息则同步保存
    candidate_counts = []          # 每个原始样本对应的候选数（用于 later repeat_interleave）

    for i in range(orig_bs):
        info_i = extra_info[i]
        model_answers = info_i.get("model_answer_id", [])
        if n_candidates is not None:
            model_answers = model_answers[:n_candidates]# 取前n个候选作为专家答案
        # record how many candidates this sample contributes
        candidate_counts.append(len(model_answers))
        for cand in model_answers:
            # cand is expected to be list[int] (未填充)
            responses_list.append(cand[:response_length]) # 这里注意截取到最大长度
            # attempt to get corresponding logprob if present in extra_info,
            # e.g., extra_info[i] might contain "model_answer_logprob" parallel structure.
            if calculate_log_probs:
                logprob_field = info_i.get("model_answer_logprob", None)
                if logprob_field is not None:
                    # logprob_field should be list of lists aligned with model_answer_id
                    # 如果存在，则把对应候选的 logprob 添加；否则填充占位（will be ignored）
                    # 防守性处理：
                    try:
                        # 找到当前 candidate 在 model_answers 中的索引
                        # 注意：如果 candidate 列表是可变对象，这里简单按顺序匹配
                        # 因为我们顺序遍历 model_answers，所以索引一致
                        pass
                    except Exception:
                        pass

    # 计算展开后的批次大小
    total_candidates = len(responses_list)
    if total_candidates == 0:
        raise ValueError("No model_answer_id candidates found in gen_batch.non_tensor_batch['extra_info']")

    # 将 responses_list 右侧 pad 到固定长度 response_length，并转为 tensor
    responses_tensor = pad_2d_list_to_length(responses_list, pad_token_id, max_length=response_length).to(device)

    # 如果需要 rollout_log_probs，尝试从 extra_info 中读取并 pad（否则不创建该字段）
    rollout_log_probs_tensor = None
    if calculate_log_probs:
        # 尝试从 extra_info 中收集 "model_answer_logprob" 字段（如果存在）
        logprob_candidates = []
        for i in range(orig_bs):
            info_i = extra_info[i]
            logprob_field = info_i.get("model_answer_logprob", None)
            model_answers = info_i.get("model_answer_id", [])
            if n_candidates is not None:
                model_answers = model_answers[:n_candidates]
                if isinstance(logprob_field, list):
                    logprob_field = logprob_field[:n_candidates]
            if logprob_field is None:
                # 填充占位（-1 的序列），以便后续 pad
                for _ in model_answers:
                    logprob_candidates.append([-1] * response_length)
            else:
                # logprob_field 应为 list[list[float]]，需要 pad/truncate 到 response_length
                for lp in logprob_field:
                    # lp 可能长度不等，这里右侧 pad -1 并截断
                    if len(lp) >= response_length:
                        clipped = lp[:response_length]
                    else:
                        clipped = lp + [-1] * (response_length - len(lp))
                    logprob_candidates.append(clipped)
        if len(logprob_candidates) != total_candidates:
            # 兜底：如果没有正确读到 logprobs，则不创建 rollout_log_probs
            rollout_log_probs_tensor = None
        else:
            rollout_log_probs_tensor = torch.tensor(logprob_candidates, dtype=torch.float32, device=device)

    # 对 prompts / attention_mask / position_ids 做 repeat_interleave，使其与 responses_tensor 对齐
    # candidate_counts 列表表示每个原始样本重复的次数（例如每个为8）
    # 我们需要将每个原始样本的行重复 candidate_counts[i] 次
    # 为此构造一个 repeats 张量/序列与原始 batch 逐行对齐
    repeats = candidate_counts  # list[int] length = orig_bs

    # 若 repeats 中有不一致（某些样本的候选数不同），我们用逐行 repeat_interleave
    # 对 torch.Tensor 使用 repeat_interleave；对 python list（non_tensor fields）使用手工展开
    # 处理 idx, attention_mask, position_ids（均为 tensor）
    # 先把 repeats 转为 torch tensor for torch.repeat_interleave
    repeats_tensor = torch.tensor(repeats, device=device, dtype=torch.long)
    # 但 torch.repeat_interleave 要求 repeats 对应每个元素的重复数是标量或与输入长度一致；
    # 我们可以使用索引 expand 法：构造 index map
    # 简单做法：对每行生成 index list 并用索引取值
    indices = []
    for i, r in enumerate(repeats):
        indices.extend([i] * r)
    indices = torch.tensor(indices, device=device, dtype=torch.long)

    idx_expanded = idx[indices]
    attention_mask_expanded = attention_mask[indices]
    position_ids_expanded = position_ids[indices]

    # 如果 non_tensor_batch 中包含需要按样本复制的字段（如 raw_prompt_ids、raw_prompt），也应展开
    non_tensor_expanded = {}
    for k, v in non_tensor.items():
        if k in non_tensor_batch_remove_keys:
            # 不存储extra_info
            continue
        else:
            # 对于其他非按样本的字段，直接原样保留
            non_tensor_expanded[k] = non_tensor[k][indices]

    
    # 现在 responses_tensor 的 shape = (total_candidates, response_length)
    # 确保 idx_expanded.shape[0] == total_candidates
    if idx_expanded.size(0) != responses_tensor.size(0):
        raise RuntimeError("Expanded prompt count does not match total number of responses")

    # 拼接 seq = prompt + response
    seq = torch.cat([idx_expanded, responses_tensor.to(device)], dim=-1)
    # breakpoint()
    # 更新 position_ids：按照 vLLM 中逻辑，response_position_ids = position_ids[..., -1:] + arange(1, response_length+1)
    batch_size_new = responses_tensor.size(0)
    delta_position_id = torch.arange(1, response_length + 1, device=device).unsqueeze(0).expand(batch_size_new, -1)
    # 若 position_ids_expanded 有 3 维的特殊结构（例如 qwen2vl mrope），按原逻辑处理
    if position_ids_expanded.dim() == 3:
        delta_position_id = delta_position_id.view(batch_size_new, 1, -1).expand(batch_size_new, position_ids_expanded.size(1), -1)
    response_position_ids = position_ids_expanded[..., -1:] + delta_position_id
    position_ids_new = torch.cat([position_ids_expanded, response_position_ids], dim=-1)

    # 生成 response_attention_mask（只对 response 部分），然后拼接 attention_mask
    response_attention_mask = get_response_mask(response_id=responses_tensor, eos_token=pad_token_id, dtype=attention_mask_expanded.dtype)
    attention_mask_new = torch.cat([attention_mask_expanded, response_attention_mask], dim=-1)

    # 构造最终的 TensorDict
    batch = TensorDict(
        {
            "prompts": idx_expanded,
            "responses": responses_tensor,
            "input_ids": seq,
            "attention_mask": attention_mask_new,
            "position_ids": position_ids_new,
        },
        batch_size=batch_size_new,
    )

    if rollout_log_probs_tensor is not None:
        batch["rollout_log_probs"] = rollout_log_probs_tensor.to(device)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_expanded)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RAFT:
        advantages, returns = core_algos.compute_raft_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn # self.expert_rool_out =  4
    
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_old_log_probs = False # 是否计算log_probs

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # basemodel_val
        self.batch_last_four_data_dict = None
        self.base_model_rollout = False
        self.base_model_compute_probs= False
        self.wait_for_compute_prob_kl_batch = None
        self.kl_batch = None
        
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.RAFT,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True): # val_reward_fn不为空，且在训练前执行val操作
            val_metrics = self._validate() # 对base_model计算初始性能指标
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)# 记录base model 的初始性能指标
            if self.config.trainer.get("val_only", False): # 只验证，不训练
                return

        # add tqdm 使用 tqdm 库创建一个进度条，用于可视化训练进度
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1 # 训练的step数
        last_val_metrics = None # 最新的val_metric指标
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        ) # 是否用sglang作为Rollout的推理框架，我们用的是vllm框架，所以忽略它

        # 获取到最后四个batch最为KL散度计算的轨迹
        _last_four_batches = deque(maxlen=4)
        for _bd in self.train_dataloader:
            _last_four_batches.append(_bd)
        # breakpoint()
        def _merge_batch_dicts(batch_dicts):
            if not batch_dicts:
                return {}
            keys = set(batch_dicts[0].keys())
            for d in batch_dicts[1:]:
                keys &= set(d.keys())
            merged = {}
            for k in keys:
                vals = [d[k] for d in batch_dicts]
                v0 = vals[0]
                if isinstance(v0, torch.Tensor):
                    merged[k] = torch.cat(vals, dim=0)
                elif isinstance(v0, np.ndarray):
                    merged[k] = np.concatenate(vals, axis=0)
                elif isinstance(v0, list):
                    tmp = []
                    for v in vals:
                        tmp.extend(v)
                    merged[k] = tmp
                elif isinstance(v0, dict):
                    merged[k] = _merge_batch_dicts(vals)
                else:
                    merged[k] = vals
            return merged

        batch_last_four_data_dict = _merge_batch_dicts(list(_last_four_batches))
        self.batch_last_four_data_dict = batch_last_four_data_dict
        kl_batch: DataProto = DataProto.from_single_dict(self.batch_last_four_data_dict)
        base_model_gen_batch = kl_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )
        
        
        
        # breakpoint()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {} # 指标记录词典
                timing_raw = {} # 耗时记录
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                
                # 问题以input_ids等形式给出，推理过程和答案在extra_info中给出
                # 模型输入必需的输入
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "extra_info"]
                
                extra_key = ["extra_info"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                ) # gen_batch是generate()所需要的数据，原始batch将这些数据pop之后只保留后续训练和优化步骤所需的数据
                # 这种分离优化了分布式传输，并确保各组件只接收必需的数据
                if repeat_sampling_sglang_grpo: # generate框架不是用的sglang,不用管
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                
                # 从专家数据集挑选数据
                # breakpoint()
                expert_batch_output = build_dataprop_from_gen_batch(gen_batch, self.config.data.max_response_length, self.tokenizer.pad_token_id, False, self.config.data.teacher_count) # 需要把4写到配置文件里去
                # 现在不需要extro_info了，把它pop掉
                gen_batch.pop(
                    non_tensor_batch_keys=extra_key,
                )
                
                # Rollout过程
                with marked_timer("step", timing_raw): # 这是一个计时逻辑，下面的操作会进行计时
                    # generate a batch Generate操作，模型根据给定的prompt生成响应
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode: # 同步模式
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            
                            # 这里顺便把basemodel的rollout结果保存下来
                            if not self.base_model_rollout:
                                gen_kl_base_model_output = self.actor_rollout_wg.generate_sequences(base_model_gen_batch)
                                kl_batch = kl_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                kl_batch = kl_batch.union(gen_kl_base_model_output)
                                kl_batch = kl_batch[torch.arange(0, len(kl_batch), step=self.config.actor_rollout_ref.rollout.n)]
                                self.wait_for_compute_prob_kl_batch = kl_batch
                                self.base_model_rollout = True
                        else:
                            # vllm should set async_rollout_mode to enable async rollout
                            # sglang turns on async_rollout_mode by default
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"]) # 将生成的耗时信息合并到timing_raw中并删除
                        gen_batch_output.meta_info.pop("timing", None)
                        
                        # breakpoint()
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not repeat_sampling_sglang_grpo:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        ) # 对原始数据生成一个uid好对应到其rollout的结果
                        # repeat to align with repeated responses in rollout
                        gen_batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # 把原始数据重复5次对应到rollout的数据
                        expert_bench = batch.repeat(repeat_times=self.config.data.teacher_count, interleave=True) # 对专家数据也重复对应次数好进行匹配

                    gen_batch_union = gen_batch.union(gen_batch_output)
                    expert_bench_union = expert_bench.union(expert_batch_output)
                    
                    # breakpoint()
                    # 合并专家数据和生成数据
                    batch = concat_dataproto_objects(gen_batch_union, expert_bench_union)
                    
                    # breakpoint() # 到这里，batch中包含了模型生成的数据和专家数据，后续会计算奖励和优势等
                    
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)# 分配数据保证所有GPU分配到的总token数尽可能接近

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"): # 计算奖励
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # breakpoint()
                    if not self.base_model_compute_probs:
                        wait_for_compute_prob_kl_batch = deepcopy(self.wait_for_compute_prob_kl_batch)
                        eval_base_old_log_prob = self.actor_rollout_wg.compute_log_prob(wait_for_compute_prob_kl_batch)
                        self.kl_batch = wait_for_compute_prob_kl_batch.union(eval_base_old_log_prob)
                        self.kl_batch.batch["response_mask"] = compute_response_mask(self.kl_batch)
                        self.base_model_compute_probs = True
                    
                    if self.config.trainer.kl_eval is not None and self.config.trainer.kl_eval == True:
                        wait_for_compute_prob_kl_batch = deepcopy(self.wait_for_compute_prob_kl_batch)
                        eval_old_log_prob = self.actor_rollout_wg.compute_log_prob(wait_for_compute_prob_kl_batch)
                        eval_kl_batch = wait_for_compute_prob_kl_batch.union(eval_old_log_prob)
                        mean_batch_kl = compute_forward_kl_divergence(self.kl_batch.batch['old_log_probs'], eval_kl_batch.batch['old_log_probs'], self.kl_batch.batch["response_mask"])
                        mean_batch_kl_metrics = {"actor/kl_eval": mean_batch_kl.detach().item()}
                        metrics.update(mean_batch_kl_metrics)
                    
                    
                    # breakpoint()
                    # recompute old_log_probs
                    # self.use_old_log_probs = True
                    if self.use_old_log_probs:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch) # 计算旧策略的对数概率
                            entropys = old_log_prob.batch["entropys"] # old model 对于token的对数概率
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )
                    else:
                        batch.meta_info['temperature'] = 1.0

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    # breakpoint()
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor # 每个token的reward

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor GRPO需要标准化

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )# 计算优势

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # breakpoint()
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch) # 更新策略模型
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
