import re
def extract_final_boxed_answer(response: str):
    # extract final boxed answer: \boxed{...}, not only true/false
    pattern = r'\boxed\{(.*?)\}'
    matches = re.findall(pattern, response)
    if matches:
        final_answer = matches[-1]
        return final_answer.strip().lower()
    return None

example = """dwadawd \boxed{True} asd \boxed{False} \boxed{True}"""
print(extract_final_boxed_answer(example))  # Output: True