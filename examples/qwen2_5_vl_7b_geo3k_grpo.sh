set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=/mnt/workspace/luoruilin.lrl/models/Qwen2-VL-2B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""User may ask Assistant to complete a problem, provide a critique of a solution, or refine an incorrect solution based on the critique content. 
 If completing a problem, you should give reasoning process and answer that are enclosed within <think> and <answer> tags, respectively. For example, <think> reasoning process here </think><answer> answer here </answer> 
 If providing a critique, you should give critique process and judgment that are enclosed within <think> and <judge> tags, respectively. For example, <think> critique process here </think><judge> judgment here (True or False) </judge> 
 If regenerating a solution, you should give correction process and answer that are enclosed within <think> and <answer> tags, respectively. For example, <think> reasoning process here </think><answer> answer here </answer>"""

wandb offline
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/workspace/luoruilin.lrl/data/geometry3k/data@train \
    data.val_files=/mnt/workspace/luoruilin.lrl/data/geometry3k/data@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=2
