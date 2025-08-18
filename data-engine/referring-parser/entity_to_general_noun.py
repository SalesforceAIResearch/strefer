import vllm
import re
import ast
from call_vllm import complete_with_vllm, load_llm_sampling_params
import sys
sys.path.append("./prompts")
from noun_gen import NOUN_GEN_PROMPT


def get_nouns_batch(model, sampling_params, entities, TOKENIZER, rep_times=3):
    entities = [str([entity]) for entity in entities]
    prompts = [NOUN_GEN_PROMPT.format(entities=entity) for entity in entities] * rep_times
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outputs = complete_with_vllm(model, sampling_params, TOKENIZER, msgs)
    return format_json(outputs, rep_times=rep_times)


def format_json(output_strings, rep_times=3):
    python_lists = []
    length = len(output_strings) // rep_times
    output_strings = [output_strings[i * length:(i + 1) * length] for i in range(rep_times)]
    for j in range(len(output_strings[0])):
        break_sign = False
        for i in range(rep_times):
            try:
                match = re.search(r"```json\s*(.*?)\s*```", output_strings[i][j], re.DOTALL)
                if match:
                    json_string = match.group(1).replace('\n', '')
                    python_list = ast.literal_eval(json_string)
                    if python_list:
                        python_lists.append(python_list)
                        break_sign = True
                        break
            except:
                pass
        if not break_sign:
            python_lists.append(["None"])

    return python_lists

if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    # model_path = "Qwen/Qwen2.5-32B-Instruct"
    MODEL, sampling_params, TOKENIZER = load_llm_sampling_params(model_path, tensor_num=1)

    entities = ['child in a white T-shirt', 'child in a pink top', 'dog with a red leash', 'woman with ponytail', 'bride', 'groom', 'bridesmaid', 'officiant']
    outputs = get_nouns_batch(MODEL, sampling_params, entities, TOKENIZER, rep_times=3)
    print(outputs)

