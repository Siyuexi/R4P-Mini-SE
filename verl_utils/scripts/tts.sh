# Scripts for evaluating Mini-SE (patch selection).

# p is part. range from 1 to 4. (4x4=16) for first comp etition
# p can also be 'x2' or 'x4' for second competition
# p=1
p=x4

export ROOT_DIR='.'
export EXPERIMENT_NAME='tts'
export FINETUNE_MODEL_PATH=$ROOT_DIR'/models/r4p'

# round-1
data_path=$ROOT_DIR/datasets/data_tts_minise_patch_x1.parquet
# round-2
data_path=$ROOT_DIR/datasets/data_tts_minise_patch_x2x4.parquet

output_path=$ROOT_DIR/datasets/output_tts_minise_$p'_'$EXPERIMENT_NAME'_'$step.parquet

# only if you need to merge model checkpoints.
# python -m verl_utils.eval.model_merger \
#   --local_dir $FINETUNE_MODEL_PATH

python3 -m verl.trainer.main_generation \
    data.batch_size=128 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt$p \
    data.n_samples=1 \
    data.output_path=$output_path \
    model.path=$FINETUNE_MODEL_PATH/huggingface \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=24576 \
    rollout.response_length=8192 \
    rollout.calculate_log_probs=true \
    rollout.max_num_batched_tokens=32768 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8 \
    2>&1 | tee $EXPERIMENT_NAME-test.log

python3 -m verl_utils.eval.tts_selector \
    --file_path=$output_path \
    --p=$p \
    --agg=random \
    2>&1 | tee $EXPERIMENT_NAME-report.log


# python3 -m verl_utils.eval.tts_selector_merge --file_patterns $ROOT_DIR/datasets/output_test_minise_true_only_1_$EXPERIMENT_NAME'_'$step.parquet $ROOT_DIR/datasets/output_test_minise_true_only_2_$EXPERIMENT_NAME'_'$step.parquet $ROOT_DIR/datasets/output_test_minise_true_only_3_$EXPERIMENT_NAME'_'$step.parquet $ROOT_DIR/datasets/output_test_minise_true_only_4_$EXPERIMENT_NAME'_'$step.parquet --agg random
