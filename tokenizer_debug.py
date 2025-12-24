"""
This file is to load the tokenizer "/mnt/weka/home/yongxin.wang/workspace/lark/models/Qwen/Qwen3-1.7B-Base"
"""

from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/weka/home/yongxin.wang/workspace/lark/models/yangzhch6/Qwen2.5-Math-7B-16k-think",
        trust_remote_code=True,
    )

    # print(tokenizer.eos_token)
    # print(tokenizer.pad_token)

    text = "<think>\nI dont know the answer.\n<step_bed619fva643c0v108hd53gcy>I don't know.</step_bed619fva643c0v108hd53gcy></think>\nI don't know the answer."
    encoded_input = tokenizer(text, return_tensors="pt")
    print("Encoded input:", encoded_input)
    decoded_output = tokenizer.decode(encoded_input["input_ids"][0])
    print("Decoded output:", decoded_output)

    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello, how are you?"}
    # ]
    # prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(prompt_str)