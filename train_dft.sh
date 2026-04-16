# train data generation
# python examples/data_preprocess/numina_cot.py --train_end 100000
# eval data generation
# python examples/data_preprocess/math_dataset.py


nproc_per_node=4
project_name=numina-cot

experiment_name=numina-cot-sft-qwen-2.5-math-1.5b
save_path=/opt/data/private/gwj/GFT/DFT/output/DFT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files=/home/gwj/ACL2026/GFT/data_preprocess/results/output.parquet \
    data.val_files=/home/gwj/ACL2026/GFT/DFT/data/test_data/processed/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=3072 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/home/Dataset/Models/Qwen/QwenMath/Qwen2.5-Math-1.5B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=1000000 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true