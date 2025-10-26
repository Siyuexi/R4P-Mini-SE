type=batch

export ROOT_DIR='/mnt/...'
export EXPERIMENT_NAME='r4p'
export FINETUNE_MODEL_PATH=$ROOT_DIR'/model/r4p'

data_path=$ROOT_DIR/datasets/data_test_r4p.parquet
output_path=$ROOT_DIR/datasets/output_r4p.parquet

# python -m verl_utils.eval.model_merger \
#   --local_dir $FINETUNE_MODEL_PATH

python3 -m verl.trainer.main_generation \
    data.batch_size=128 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$output_path \
    model.path=$FINETUNE_MODEL_PATH/huggingface \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=24576 \
    rollout.response_length=8192 \
    rollout.max_num_batched_tokens=32768 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.7 \
    2>&1 | tee $EXPERIMENT_NAME-test.log

python3 -m verl_utils.eval.result_evaluator \
    --file_path=$output_path \
    --type=batch \
    2>&1 | tee $EXPERIMENT_NAME-report.log