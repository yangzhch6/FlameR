# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional, List

import requests
import random
import re
import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score
from verl.workers.reward_manager import register



async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=60.0) for c, r, t, ei in zip(completions, references, tasks, extra_info)]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            # print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, (int, float, bool)):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes))
    finally:
        loop.close()


def math_verify_score(task: str, model_output: str, ground_truth: str, extra_info: Optional[dict] = None):
    if extract_final_boxed_answer(model_output) == "":
        return 0.0
    return math_verify_compute_score(model_output[-300:], ground_truth)


def math_verify_score_think(task: str, model_output: str, ground_truth: str, extra_info: Optional[dict] = None):
    if model_output.count("</think>") == 0:
        return 0.0

    model_output = model_output.split("</think>")[-1].strip()
    if extract_final_boxed_answer(model_output) == "":
        return 0.0
    return math_verify_compute_score(model_output[-300:], ground_truth)


# SYS_VERIFY_CAL = """You are a mathematical solution evaluator. Your task is to analyze whether a given solution correctly answers a given mathematical question.  

# Follow these steps in your response:  
# 1. Read the mathematical question carefully.
# 2. Go through the provided solution step by step.
# 3. Check for logical consistency, mathematical correctness, and relevance to the question.
# 4. Finally, output a binary judgement in the exact format:
#    - If the solution is fully correct, write: `\\boxed{True}`
#    - If the solution is incorrect or partially incorrect, write: `\\boxed{False}`

# Do not add extra text after the boxed judgement."""

SYS_VERIFY_CAL= """You are a mathematical solution evaluator. Your task is to analyze whether the final answer in a given solution correctly answers a given mathematical question.

Note: The provided solution may be only the tail part of a full solution. Your focus should be on verifying the final answer, which is enclosed in \\boxed{}.

Follow these steps in your response:
1. Read the mathematical question carefully.
2. Analyze the provided solution part. Check the final answer, which is usually enclosed in `\\boxed{}`.
3. Reason step by step to verify whether the final answer is correct for the mathematical question. Check for logical consistency and mathematical correctness based on the context provided.
4. Finally, output a binary judgement in the exact format:
   - If the final answer is correct, write: `\\boxed{True}`
   - Otherwise (including if the answer is incorrect or missing), write: `\\boxed{False}`
Do not add extra text after the boxed judgement."""


# SYS_VERIFY_CAL = """You are a mathematical solution evaluator. Your task is to analyze whether the final answer in a given solution correctly answers a given mathematical question.  

# Note: The provided solution may be only the tail part of a full solution. Your focus should be on verifying the final answer, which is enclosed in \\boxed{}.

# Follow these steps in your response:
# 1. Read the mathematical question carefully.
# 2. Examine the provided solution. If the solution does not contain a \\boxed{} with a final answer, immediately judge it as incorrect.
# 3. If there is a \\boxed{} with a final answer, check if this answer is mathematically correct and relevant to the question. Consider the context of the solution, even if it's only a tail part.
# 4. Finally, output a binary judgement in the exact format:
#    - If the final answer in \\boxed{} is correct, write: \\boxed{True}
#    - Otherwise, or if no final answer is provided, write: \\boxed{False}

# Do not add extra text after the boxed judgement.
# """

USER_VERIFY_CAL = """# Question:
{question}

# Solution:
```
{solution}
```"""


def extract_final_boxed_answer(response: str):
    # extract final boxed answer: \boxed{...}, not only true/false
    pattern = r'boxed\{(.*?)\}'
    matches = re.findall(pattern, response)
    if matches:
        final_answer = matches[-1]
        return final_answer.strip().lower()
    return ""

def single_rm_compute(question: str, solution: str, rm_server_ip: str, check_think=True):
    if check_think:
        if solution.count("</think>") == 0:
            return False
        solution = solution.split("</think>")[-1].strip()

    if len(solution) > 600:
        solution = "...\n" + solution[-600:]  # 截断，避免过长

    if extract_final_boxed_answer(solution) == "":
        return 0.0

    """
    单个样本请求 RM 的逻辑
    """
    ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
    select_port = random.choice(ports)
    
    chat_url = f"http://{rm_server_ip}:{select_port}/v1/chat/completions"
    
    request_message = [
        {"role": "system", "content": SYS_VERIFY_CAL},
        {"role": "user", "content": USER_VERIFY_CAL.replace("{question}", question).replace("{solution}", solution)},
    ]

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "RewardModel",
        "messages": request_message,
        "max_tokens": 4096,
        "temperature": 0.6,
        "n": 5 # 保持你的设置，采样5次
    }

    max_retries = 6
    retry_count = 0
    rm_results = None

    # 1. 获取网络结果
    while retry_count < max_retries:
        try:
            # 随机选择端口重试，避免单点故障
            current_port = random.choice(ports) 
            chat_url = f"http://{rm_server_ip}:{current_port}/v1/chat/completions"
            
            response = requests.post(chat_url, json=data, headers=headers, timeout=120)
            
            if response.status_code == 200:
                rm_results = response.json()
                
                if random.random() < 0.01: # 1% 概率打印输出，避免刷屏
                    # print(f"\033[94m[Solution]\033[0m {solution}")
                    first_response = rm_results["choices"][0]["message"]["content"]
                    print(first_response)
                    print("="*60)
                break # 成功
            else:
                # print(f"请求失败，状态码: {response.status_code}，第{retry_count + 1}次尝试")
                pass
        except Exception as e:
            # print(f"请求异常: {e}，第{retry_count + 1}次尝试")
            pass
        
        retry_count += 1
        if retry_count < max_retries:
            time.sleep(1) # 稍微sleep一下，避免过于密集
    
    if rm_results is None:
        return 0.0 # 所有重试都失败
    
    # 2. 解析分数
    scores = []
    if "choices" in rm_results:
        for choice in rm_results["choices"]:
            response_content = choice["message"]["content"]
            final_answer = extract_final_boxed_answer(response_content)
            if final_answer == "true":
                scores.append(1.0)
            elif final_answer == "false":
                scores.append(0.0)
            else:
                continue
    
    if len(scores) == 0:
        return 0.0

    avg_score = sum(scores) / len(scores)
    return avg_score

def run_parallel_rm_scoring(questions, solutions, rm_server_ip, check_think=True, num_workers=64):
    """
    并行运行 RM 打分 (使用 ThreadPool)
    """
    scores = []
    # 使用 ThreadPoolExecutor，因为 requests 是 IO 阻塞的
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 固定 rm_server_ip 参数
        func = partial(single_rm_compute, rm_server_ip=rm_server_ip, check_think=check_think)
        # 并行执行
        results = executor.map(func, questions, solutions)
        scores = list(results)
    return scores


# ============================================================================
#  Manager Class
# ============================================================================

class RMMathRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    Modified to support both MathVerify and Model-based Reward (RM).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        rm_server_ip: str = "fs-mbz-gpu-758", # 新增参数
        check_think_format: bool = False,
        reward_fn_key: str = "data_source",
        threshold: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.check_think_format = check_think_format
        self.rm_server_ip = rm_server_ip # 保存 IP
        self.threshold = threshold

        if self.check_think_format:
            self.compute_score = math_verify_score_think
        else:
            self.compute_score = math_verify_score

    def verify_math(self, sequences_str, ground_truth, data_sources, extra_info):
        """
        计算 Math Verify 分数 (Rule-based)
        """
        try:
            # 预热/测试单条
            if len(sequences_str) > 0:
                single_test = self.compute_score(data_sources[0], sequences_str[0], ground_truth[0], extra_info[0])
                print("# success test single sample...")

            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64, 
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global math reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during math scoring. Setting all as 0. \n##Info: {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        
        return scores

    def verify_rm(self, prompt_str, sequences_str):
        """
        计算 Reward Model 分数 (Model-based)
        """
        try:
            scores = run_parallel_rm_scoring(
                questions=prompt_str,
                solutions=sequences_str,
                rm_server_ip=self.rm_server_ip,
                check_think=self.check_think_format,
                num_workers=64 # IO 密集型任务使用线程池，可以开大一点
            )
        except Exception as e:
            print(f"[Error] Unexpected error during RM scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        
        scores = [1.0 if s >= self.threshold else 0.0 for s in scores]
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Orchestrates both Math Verification and Reward Model scoring.
        """

        # If there is rm score, we directly return rm score.
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # 1. Decode Inputs
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        extra_info = data.non_tensor_batch.get("extra_info", None)
        questions = [extra["question"] for extra in extra_info]

        assert len(sequences_str) == len(ground_truth) == len(data_sources) == len(extra_info)
        # 2. Compute Math Verify Scores (Rule-based)
        # 结果放入 data.batch["math_verify_acc"]
        print("## Computing math-verify reward...")
        math_scores = self.verify_math(sequences_str, ground_truth, data_sources, extra_info)
        data.batch["math_verify_acc"] = torch.tensor(math_scores, dtype=torch.float32, device=prompt_ids.device)

        # 3. Compute Reward Model Scores (Network-based)
        # 结果放入 data.batch["acc"]
        # RM 需要输入 Question (Prompt) 和 Solution (Response)
        print("## Computing reward-model rewards...")
        rm_scores = self.verify_rm(questions, sequences_str)
        data.batch["acc"] = torch.tensor(rm_scores, dtype=torch.float32, device=prompt_ids.device)

        # 4. Construct Reward Tensor (Using RM Scores as the primary training signal)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_source = data_sources[i]
            # 这里使用 rm_scores 来填充 reward_tensor，因为你的需求是训练使用 RM
            reward_tensor[i, valid_response_length[i].item() - 1] = rm_scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"<<<<<<<<<<  Prompt-{i}  >>>>>>>>>>:\n{prompt_str[i][-500:]}...") # 打印最后部分避免刷屏
                print(f"<<<<<<<<<< Response-{i} >>>>>>>>>>:\n{sequences_str[i][-500:]}...")
                # 打印两种分数
                print(f"<<<<<<<<<< Evaluate-{i} >>>>>>>>>>:\nGT: {ground_truth[i]} | MathVerify: {math_scores[i]} | RM Score: {rm_scores[i]} | Data Source: {data_source}")

        if return_dict:
            return rm_scores, {"reward_tensor": reward_tensor}
        else:
            return rm_scores, reward_tensor