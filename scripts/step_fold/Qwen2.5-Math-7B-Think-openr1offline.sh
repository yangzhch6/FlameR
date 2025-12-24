set -x
export WANDB_API_KEY="004ba186f7e1f9bd08fe620ddeaaf98ef356c95f"
export HF_TOKEN="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"
export RAY_BACKEND_LOG_LEVEL=error
export NUM_NODES=1
export NUM_GPUS=8

## model, file and save path 
project_name='Folding_Thought'
experiment_name='Qwen2.5-Math-7B-16k-Openr1-Prompt2-Step-Fold-SFT-r12k'
model_name_or_path=/mnt/weka/home/yongxin.wang/workspace/lark/swift-pipeline/ckpt/think-step/Qwen2.5-Math-7B-16k-think-Openr1-Prompt2-Step-Fold/v0-20251222-131235/checkpoint-5493 # path to the pre-trained model
save_path=/mnt/weka/home/yongxin.wang/workspace/lark/SvS/checkpoints/${project_name}/${experiment_name} # define the path for saving RL intermediate checkpoints

## data config
train_path=data/think-step-fold/openr1-math-46k-qwen25math.parquet
test_path=data/think-step-fold/aime2425_amc23_math500_minerva_qwen25math.parquet

## system parameters
use_chat_template=False
val_before_train=True
use_dynamic_bsz=True
tensor_model_parallel_size=1 # rollout and training batch size
use_tqdm=True # whether using tqdm in vLLM generation
save_freq=100
test_freq=100

## training parameters
check_think_format=True
check_step_format=True
total_epochs=30
total_training_steps=500
train_batch_size=128
ppo_mini_batch_size=64
log_prob_micro_batch_size_per_gpu=2
kl_coef=0.0
kl_loss_coef=0.0
learning_rate=1e-6
temperature=0.6
n_samples=8
max_prompt_length=4096 
max_response_length=12288
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
estimator=grpo
use_kl_loss=$( [ "$(echo "$kl_loss_coef > 0.0" | bc)" -eq 1 ] && echo true || echo false )
use_kl_in_reward=$( [ "$(echo "$kl_coef > 0.0" | bc)" -eq 1 ] && echo true || echo false )


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${estimator} \
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.check_think_format=${check_think_format} \
    +data.check_step_format=${check_step_format} \
    data.use_chat_template=${use_chat_template} \
    actor_rollout_ref.model.path=${model_name_or_path} \
    +actor_rollout_ref.ref.model.path=${model_name_or_path} \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=k2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.n=${n_samples} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=${n_samples} \
    actor_rollout_ref.rollout.use_tqdm=${use_tqdm} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$((log_prob_micro_batch_size_per_gpu * 2)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.critic_warmup=0 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.nnodes=${NUM_NODES} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq}  \
    trainer.val_only=${val_only} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.val_before_train=${val_before_train} \
    trainer.default_local_dir=${save_path} \
    trainer.task='rl' $@ # set the task to 'rl' or 'svs'