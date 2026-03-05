import pandas as pd
from datasets import concatenate_datasets, load_dataset

SYS_PROMPT_THINK = """Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. 

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.

After “</think>\n” in the Solution section, provide the final, logical, and accurate solution, clearly derived from the exploration in the Thought section.

If applicable, include the Final Answer in \\boxed{} for closed-form results like multiple choices or mathematical answers."""


SYS_PROMPT_REF1 = """You are a math AI assistant. You need to solve the given math problem. For calculation problem, show your work clearly and put your final answer within \\boxed{}. For proof problem, provide a rigorous logical derivation. Ensure your solution is clearly stated. (You can only use natural language, not formal language.)\n\nYou will be provided with a Question and its Ground Truth Solution. In order to guarantee accuracy, you should consult the ground truth to inform your thought process and your own solution. However, it is imperative that you **NEVER** reference or suggest the existence of a \"ground truth\" in your thought process or final solution. **You must solve/prove the problem as if you are reasoning from scratch, solely relying on your own abilities.**"""



SYS_PROMPT_REF2 = """You are a math AI assistant. You need to solve the given math problem. Show your work clearly and put your final answer within \\boxed{}. 

You will be provided with a Question and its Ground Truth Solution. In order to guarantee accuracy, you should consult the ground truth to inform your thought process and your own solution. However, it is imperative that you **NEVER** reference or suggest the existence of a \"ground truth\" in your thought process or final solution. **Pretend you are reasoning from scratch, relying solely on your own abilities to solve/prove the problem.**"""



# load data from huggingface Elliott/Openr1-Math-46k-8192
train_dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

formated_dataset = []
for line in train_dataset:
    question = line['prompt'][1]['content'].strip()
    thought_solution = line["target"][0]['content'].strip()
    solution = line["target"][0]['content'].split("</think>")[-1].strip()
    formated_dataset.append({
        "prompt": [{"role":"system", "content": SYS_PROMPT_REF2}, {"role": "user", "content": "## *Question*:\n{question}\n\n\n\n## *Ground Truth Solution*:\n{solution}".format(question=question, solution=solution)}],
        "gt_output": thought_solution,
        "data_source": "Openr1-Math-46k",
        "reward_key": "DAPO-17k",
        "ability": 'math',
        "reward_model": line['reward_model'],
        "extra_info": {
            "solution": solution,
            "index": len(formated_dataset),
            "split": "train"
        },
    })

print(formated_dataset[0])
print(f"Total samples: {len(formated_dataset)}")


# save as parquet
df = pd.DataFrame(formated_dataset)
df.to_parquet("/mnt/weka/home/yongxin.wang/workspace/lark/verl-gan/data/thought-syn-gan/openr1-math-46k-ref2.parquet", index=False)
