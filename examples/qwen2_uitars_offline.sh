set -x

# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

# MODEL_PATH=../Lyra/work_dirs/Steve_uitars_7B_128k_sft0218_fullcorrect_with_single_frame_from_grounding_hf

# MODEL_PATH=/gpfs/lufanbin/projects/Qwen2-VL-Finetune/work_dirs/steve_uitars_all_data0405_epoch2
MODEL_PATH=/gpfs/lufanbin/projects/Qwen2-VL-Finetune/work_dirs/steve_uitars_all_data0405_fiximage_epoch2
MODEL_PATH=/gpfs/lufanbin/models/ui-tars/UI-TARS-1.5-7B
# MODEL_PATH=/gpfs/pretrain-models/qwen2.5-vl/Qwen2.5-VL-3B-Instruct/
# GROUNDING_MODEL_PATH=/gpfs/lufanbin/models/ui-tars/UI-TARS-7B-DPO

SYSTEM_PROMPT="""You are helpful assistant."""
    # data.max_prompt_length=90000 \
    # data.max_response_length=4096 \

NUM_GPUS=8
NUM_ENVS=32
ROLLOUT_N=4

((ROLLOUT_BSZ = NUM_ENVS/ROLLOUT_N))

    # data.train_files=data/evaluation_examples/test_evaluation.json \
    # data.train_files=data/evaluation_examples/test_success.json \
    # data.train_files=data/evaluation_examples/test_all.json \

    # rollout batch_size = 32 = rollout_n * actor.global_batch_size

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=data/evaluation_examples/test_success.json \
    data.val_files=data/evaluation_examples/test_success.json \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=$NUM_ENVS \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.ppo_epochs=2 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=$ROLLOUT_BSZ \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.grounding_model_path=${GROUNDING_MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.temperature=0.8 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.limit_images=15 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=128000 \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    trainer.experiment_name=osworld_cot_7b_v11_bsz32_n4 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.total_episodes=2 
