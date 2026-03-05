python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/weka/home/yongxin.wang/workspace/lark/Folding-Thoughts/checkpoints/Accordion-Thinking/Qwen3-4B-Base-Openr1MATH46KStepFold-mix-d6r6k-Unfold16k/global_step_300/actor \
    --target_dir /mnt/weka/home/yongxin.wang/workspace/lark/Folding-Thoughts/checkpoints/Accordion-Thinking/Qwen3-4B-Base-Openr1MATH46KStepFold-mix-d6r6k-Unfold16k/global_step_300/actor_hf