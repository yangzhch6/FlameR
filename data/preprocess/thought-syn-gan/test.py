import re
import pandas as pd
from datasets import concatenate_datasets, load_dataset

SYS_PROMPT_THINK = """Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. 

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.

After “</think>\n” in the Solution section, provide the final, logical, and accurate solution, clearly derived from the exploration in the Thought section.

If applicable, include the Final Answer in \\boxed{} for closed-form results like multiple choices or mathematical answers."""



SYS_PROMPT_REF1 = """You are a math AI assistant. You need to solve the given math problem. For calculation problem, show your work clearly and put your final answer within \\boxed{}. For proof problem, provide a rigorous logical derivation. Ensure your solution is clearly stated. (You can only use natural language, not formal language.)\n\nYou will be provided with a Question and its Ground Truth Solution. In order to guarantee accuracy, you should consult the ground truth to inform your thought process and your own solution. However, it is imperative that you **NEVER** reference or suggest the existence of a \"ground truth\" in your thought process or final solution. **You must solve/prove the problem as if you are reasoning from scratch, solely relying on your own abilities.**"""



SYS_PROMPT_REF2 = """You are a math AI assistant. You need to solve the given math problem. Show your work clearly and put your final answer within \\boxed{}. 

You will be provided with a Question and its Ground Truth Solution. In order to guarantee accuracy, you should consult the ground truth to inform your thought process and your own solution. However, it is imperative that you **NEVER** reference or suggest the existence of a \"ground truth\" in your thought process or final solution. **Pretend you are reasoning from scratch, relying solely on your own abilities to solve/prove the problem.**"""



def extract_last_boxed(text):
    """
    抽取文本中最后一个 boxed{} 中的内容。
    
    参数:
    text (str): 输入的文本
    
    返回:
    str or None: 最后一个 boxed{} 中的内容，如果没有找到则返回 None
    """
    # 使用正则表达式匹配所有 boxed{} 内容
    # .*? 是非贪婪匹配，确保只匹配到最近的 }
    matches = re.findall(r'boxed\{((?:[^{}]|{[^{}]*})*)\}', text)
    
    # 如果找到了匹配项，返回最后一个
    if matches:
        return matches[-1]
    else:
        return None

def preprocess_fn(example, idx, source):
    return {
        "question": example["problem"] if "problem" in example else example["question"],
        "ground_truth": example["answer"] if "answer" in example else example["solution"],
        "data_source": source,
    }

amc23 = load_dataset("math-ai/amc23", split="test")
aime24 = load_dataset("math-ai/aime24", split="test")
aime25 = load_dataset("math-ai/aime25", split="test")
math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
minerva = load_dataset("math-ai/minervamath", split="test")

# # train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
amc23 = amc23.map(preprocess_fn, with_indices=True, fn_kwargs={"source": "AMC23"})
aime24 = aime24.map(preprocess_fn, with_indices=True, fn_kwargs={"source": "AIME24"})
aime25 = aime25.map(preprocess_fn, with_indices=True, fn_kwargs={"source": "AIME25"})
math500 = math500.map(preprocess_fn, with_indices=True, fn_kwargs={"source": "MATH-500"})
minerva = minerva.map(preprocess_fn, with_indices=True, fn_kwargs={"source": "MINERVA"})


formated_dataset = []
for dataset in [amc23, aime24, aime25, math500, minerva]:
    for line in dataset:
        question = line['question'].strip()
        ground_truth = line["ground_truth"].strip()
        if "boxed" in ground_truth:
            print(ground_truth)
            ground_truth = extract_last_boxed(ground_truth)
            print(ground_truth)
            assert ground_truth is not None

        data_source = line["data_source"]
        formated_dataset.append({
            "prompt": [{"role":"system", "content": SYS_PROMPT_REF2},{"role": "user", "content": question}],
            "data_source": line['data_source'],
            "reward_key": "MATH-500",
            "ability": 'math',
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": {
                "answer": "\\boxed{" + ground_truth + "}",
                "index": len(formated_dataset),
                "split": "test"
            },
        })

print(formated_dataset[0])
print(f"Total samples: {len(formated_dataset)}")

# save as parquet
df = pd.DataFrame(formated_dataset)
df.to_parquet("/mnt/weka/home/yongxin.wang/workspace/lark/verl-gan/data/thought-syn-gan/amc23_aime2425_math500_minerva.parquet", index=False)


# random select 20 as debug
import random
random.seed(42)
debug_samples = random.sample(formated_dataset, 20)
df = pd.DataFrame(debug_samples)
df.to_parquet("/mnt/weka/home/yongxin.wang/workspace/lark/verl-gan/data/thought-syn-gan/debug.parquet", index=False)