import re
import json
import time
import torch
import random
import requests
from pathlib import Path
from typing import Tuple, List, Any, Optional, Callable, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.utils.generic import ModelOutput
from transformers import BitsAndBytesConfig
from utils.utils import Usage
from datetime import datetime


class ModelConfiguration:
    def __init__(self, model, config_name, value, enable=True):
        self.model = model
        self.config_name = config_name

        self.old_value = model.get_generation_config(config_name)
        self.new_value = value

        self.enable = enable

    def __enter__(self):
        if self.enable:
            self.model.set_generation_config(self.config_name, self.new_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:  # recover the original setting
            self.model.set_generation_config(self.config_name, self.old_value)


class LLM:
    def __init__(self, kwargs: Dict):
        super().__init__()

        self.max_attempts = max(kwargs.get('max_attempts', 5), 1)
        self.usage = Usage()

        match = re.match(r'^[a-zA-Z]+', kwargs['model'].rstrip("/").split("/")[-1])
        default_genre = match.group(0).lower() if match else "llm"
        self.genre = kwargs.get("genre", None) or default_genre

        self.model_name = kwargs['model']
        self.generation_config = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 8192}
        self.generation_config.update({key: kwargs[key] for key in
                                       ("do_sample", "temperature", "max_new_tokens", "top_p", "top_k", "seed")
                                       if kwargs.get(key) is not None})
        self.model = None

    def get_generation_config(self, key: str):
        return self.generation_config.get(key, None)

    def set_generation_config(self, key: str, value: Any):
        if value is None: return
        self.generation_config[key] = value

    def inner_understand(self,
                         query: str,
                         history: List[Tuple[str, str]],
                         targets: Optional[List[str]],
                         **kwargs) -> Tuple[torch.Tensor, str, Any, Usage]:
        raise NotImplementedError()

    def understand(self,
                   query: str,
                   history: List[Tuple[str, str]] = None,
                   targets: Optional[List[str]] = None,
                   post_process: Optional[Callable[[torch.Tensor, str, ModelOutput], Tuple]] = None,
                   verbose: bool = False,
                   **kwargs) -> Tuple[torch.Tensor, str, ModelOutput, Usage]:
        if history is None: history = []

        if verbose:
            print("\n#*" * 10 + " Query " + "*#" * 10 + "\n" + query.replace("\n\n", "\n"))

        # retry if raise errors
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # Retries keep the same decoding policy; never mutate shared config.
                start_timestamp = datetime.now()

                prob, response, outputs, usage = self.inner_understand(query, history, targets, **kwargs)

                duration = max((datetime.now() - start_timestamp).total_seconds(), 1e-9)
                print(f"Gen {usage['completion_tokens']} tokens in {round(duration, 1)}.", end=" ")
                print(f"{round(usage['completion_tokens'] / duration, 1)} tokens/s")

                if verbose:
                    print("\n#*" * 10 + " Response " + "*#" * 10 + "\n" + response.replace("\n\n", "\n"))

                # output check
                if post_process: prob, response, outputs = post_process(prob, response, outputs)
                break

            except Exception as e:
                attempts += 1
                time.sleep(random.randint(0, 10))
                print("Error occurs", e, f"Retrying {attempts + 1}/{self.max_attempts}")
                if attempts == self.max_attempts: raise e

        # record usage
        self.usage += usage

        return prob, response, outputs, usage


class LocalModel(LLM):
    def __init__(self, kwargs: Dict):
        super().__init__(kwargs)
        if kwargs.get("max_new_tokens") is None:
            self.generation_config["max_new_tokens"] = 16384

        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(kwargs["model"],
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          dtype='auto')

    def inner_understand(self, query, history, targets, **kwargs):
        messages = []
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        qwen3_think = kwargs.pop('think', False)  # for qwen3 mixed inference only
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=qwen3_think)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generation = {**self.generation_config, **kwargs}
        generation.pop("seed", None)  # Transformers uses the externally seeded RNG.
        if not generation.get("do_sample", False):
            generation.pop("temperature", None)  # greedy decoding in Transformers
        outputs = self.model.generate(**model_inputs, return_dict_in_generate=True, **generation)

        generated_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")

        usage = Usage({"n_completions": 1,
                       "completion_tokens": len(generated_ids),
                       "prompt_tokens": len(model_inputs.input_ids[0]),
                       "total_tokens": len(outputs.sequences[0]),
                       })

        return None, response, outputs, usage


class LocalVLLMModel(LLM):
    def __init__(self, kwargs: Dict):
        from vllm import LLM as VLLM
        super().__init__(kwargs)

        self.config = AutoConfig.from_pretrained(kwargs["model"], trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model"], trust_remote_code=True)

        self.model = VLLM(
            model=kwargs["model"],
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",
            gpu_memory_utilization=0.9,
            max_model_len=65536,
            enable_prefix_caching=True)

    def inner_understand(self, query, history, targets, **kwargs):
        from vllm import SamplingParams

        messages = []
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        qwen3_think = kwargs.pop('think', False)  # for qwen3 mixed inference only
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=qwen3_think)

        generation = {**self.generation_config, **kwargs}
        sampling_params = SamplingParams(
            temperature=float(generation.get('temperature', 0.0)) if generation.get('do_sample', False) else 0.0,
            max_tokens=int(generation.get('max_new_tokens', 8192)),
            top_p=float(generation.get('top_p', 1.0)),
            top_k=int(generation.get('top_k', -1)),
            seed=generation.get('seed'), skip_special_tokens=True)

        outputs = self.model.generate(text, sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        prompt_ids = self.tokenizer.encode(text)
        generated_ids = outputs[0].outputs[0].token_ids

        usage = Usage({"n_completions": 1,
                       "completion_tokens": len(generated_ids),
                       "prompt_tokens": len(prompt_ids),
                       "total_tokens": len(prompt_ids) + len(generated_ids),
                       })

        return None, response, outputs, usage


class APIModel(LLM):
    def __init__(self, args: Dict):
        super().__init__(args)

        self.model_name = args['model']
        self.server = args['server']
        self.thinking = args.get("thinking", False)

        api_info = json.loads(Path(f'./utils/apis/{self.server}.key').read_text())
        self.key = api_info["key"]
        self.url = api_info["url"]
        self.client = None

    def _prepare_generation_kwargs(self, targets, kwargs):
        params = dict(kwargs)
        generation = dict(self.generation_config)
        for key in ("do_sample", "temperature", "max_new_tokens", "top_p", "top_k", "seed"):
            if key in params:
                generation[key] = params.pop(key)
        params['temperature'] = float(generation.get('temperature', 0.0)) if generation.get('do_sample', False) else 0.0
        params.setdefault('max_tokens', int(generation.get('max_new_tokens', 8192)))
        for key in ("top_p", "seed"):
            if key in generation:
                params[key] = generation[key]
        if targets:
            params['top_logprobs'] = len(targets) * 2
        return params

    def inner_understand(self, query, history, targets, **kwargs):
        messages = []
        if 'system' in kwargs:
            messages.append({"role": "system", "content": kwargs.pop('system')})
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        if isinstance(self.key, list):
            key = random.choice(self.key)
        else:
            key = self.key

        kwargs = self._prepare_generation_kwargs(targets, kwargs)

        if "qwen3" in self.model_name.lower():
            kwargs["extendParams"] = {
                "enableThinking": False  # for qwen3
            }

        output = requests.post(self.url + "/chat/completions",
                               json={"model": self.model_name,
                                     "messages": messages,
                                     **kwargs},
                               headers={"Authorization": f'Bearer {key}'})

        if output.status_code != 200:
            raise ValueError(output.json())

        # phrase outputs
        output = output.json()
        response = output['choices'][0]['message']['content']
        usage = Usage({"n_completions": 1,
                       "completion_tokens": output['usage']['completion_tokens'],
                       "prompt_tokens": output['usage']['prompt_tokens'],
                       "total_tokens": output['usage']['total_tokens'],
                       })

        output["request_config"] = {"model": self.model_name, "generation": kwargs}
        return None, response, output, usage


def load_model(kwargs: Dict):
    server = kwargs.get('server', None)
    if server in [None, "", "local"]:
        return LocalModel(kwargs)
    elif server in ['vllm']:
        return LocalVLLMModel(kwargs)
    else:
        return APIModel(kwargs)
