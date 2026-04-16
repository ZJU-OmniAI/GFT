#!/bin/bash
set -x

# 1. 强制使用 vLLM 稳定版引擎 (防止 V1 实验版导致的死锁)
export VLLM_USE_V1=0
# 2. 增加 Ray 的容忍时间 (防止因为生成数据太慢，Ray 以为进程死了把它杀掉)
export RAY_task_lease_timeout_ms=1000000  # 1000秒
export RAY_object_store_idle_timeout_ms=1000000
export WANDB_MODE="offline"
# 3. 让 NCCL 报错而不是卡死 (如果有通信问题，会直接抛出 Error 而不是无限等待)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 新增：解决显存碎片化问题
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_path=/home/gwj/ACL2026/GFT/data_preprocess/results/output_50000.parquet
test_path=/home/gwj/ACL2026/GFT/DFT/data/test_data/processed/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.reward_fn_key="source" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.teacher_count=4 \
    +data.student_count=4 \
    actor_rollout_ref.model.path=/home/gwj/ACL2026/models/BaseModel/deepseek-math-7b-base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    algorithm.use_kl_in_reward=False \
    +actor_rollout_ref.actor.policy_loss.tao=0.3 \
    custom_reward_function.path=/home/gwj/ACL2026/GFT/DFT/verl/verl/utils/reward_score/NuminaMath.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='GFT' \
    trainer.experiment_name='Llama-3b' \
    trainer.default_local_dir=/home/gwj/ACL2026/GFT/DFT/outputs/GFT_llama_3B_2 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5000 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@

# trainer.resume_mode=resume_path \
# trainer.resume_from_path=/home/gwj/ACL2026/GFT/DFT/outputs/GFT/global_step_50 \