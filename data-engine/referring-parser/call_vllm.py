import vllm

def make_conv(msgs, tokenizer):
    out = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in msgs]
    return out

def load_llm_sampling_params(model_path, tensor_num=1):
    model = vllm.LLM(
        model_path,
        max_model_len=10000,
        enable_prefix_caching=True,
        load_format='safetensors',
        tensor_parallel_size=tensor_num,
    )
    tokenizer = model.get_tokenizer()
    sampling_params = vllm.SamplingParams(
        temperature=0.85, top_p=0.9, max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    return model, sampling_params, tokenizer

def complete_with_vllm(model, sampling_params, tokenizer, msgs):
    prompts = make_conv(msgs, tokenizer)
    outputs = model.generate(prompts, sampling_params, use_tqdm=True)
    completions = [output.outputs[0].text for output in outputs]
    return completions
