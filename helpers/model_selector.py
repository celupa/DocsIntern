from pathlib import Path
import sys
import traceback
import gc
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from helpers import custom_functions as cfoos


class ModelSelector():
    """Choses a model based on the available hardware (CPU/GPU)"""

    def __init__(self, model_config: dict, device: str=None):
        try:
            # instance variable
            self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
            self.do_sample = model_config["do_sample"]
            self.temperature = model_config["temperature"]
            self.top_k = model_config["top_k"]
            self.top_p = model_config["top_p"]
            self.tokenizer_path = model_config["tokenizer_path"]
            self.gpu_model_path = model_config["gpu_model_path"]
            self.cpu_model_path = model_config["cpu_model_path"]
            self.emb_model_path = model_config["emb_model_path"]
            self.model_config = model_config
            self.model_selection = {
                "cuda": (
                    self.gpu_model_path, 
                    lambda model_path: ModelSelector.load_gpu_model(model_path),
                    lambda messages: self._prompt_gpu_llm(messages)
                    ),
                "cpu": (
                    self.cpu_model_path, 
                    lambda model_path: ModelSelector.load_cpu_model(model_path),
                    lambda messages: self._prompt_cpu_llm(messages)
                    )
            }
            # model variables
            self.messages = []
            self.system_prompt = model_config["system_prompt"]
            self.model_path = self.model_selection[self.device][0]
            self.model_name = self.model_path.name
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token
            self.model = self.model_selection[self.device][1](self.model_path)
            prompt_args = ["temperature", "top_k", "top_p"]
            self.prompt_args = {k: v for k, v in self.model_config.items() if k in prompt_args}
            # print(f"---{datetime.now()} | Loaded: {self.model_name} ({self.device.upper()})")
            cfoos.LOG.info(f"Loaded: {self.model_name} ({self.device.upper()})")
        except Exception:
            # print(f"---{datetime.now()} | Failed to load model. Please check model config.")
            cfoos.LOG.info("Failed to load model. Please check model config.")
            traceback.print_exc()
            sys.exit()

    @staticmethod
    def load_gpu_model(gpu_model_path: Path) -> AutoModelForCausalLM:
        """Load GPU quantized model."""
        return AutoModelForCausalLM.from_pretrained(gpu_model_path)

    # TODO: too specific, need a model-agnostic loader here
    @staticmethod
    def load_cpu_model(cpu_model_path: Path) -> Llama:
        """Load CPU quantized model."""
        return Llama(model_path=str(cpu_model_path))

    def get_messages(self, prompt: str) -> list[dict]:
        """Build expected format for conversation history."""
        # build current conversation history
        # if conversation history is empty append system and user prompt
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": self.system_prompt})
            self.messages.append({"role": "user", "content": prompt})
        # if conversation is already ongoing, store only user prompt
        else:
            self.messages.append({"role": "user", "content": prompt})
        return self.messages
    
    def get_singleton(self, prompt: str, sysprompt: str=None) -> list[dict]:
        """
        Return an isolated conversation (system-user binomial).
        We use this mechanic to run the model independently of the main application.
        For instance, for testing or function requiring one shot evaluations.
        params:
            extend: if true, will append response to existing conversation only
        """
        sysprompt = sysprompt if sysprompt else self.system_prompt
        messages = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    # def stringify_messages(self, messages: list[dict]) -> str:
    #     """Convert messages format into a string."""
    #     return "\n".join(str(mess) for mess in messages).replace("{", "").replace("}", "")

    def extend_conversation(self, prompt: str, llm_response: str) -> None:
        """Extend conversation (self.messages) with user prompt and llm response"""
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": self.system_prompt})
        if "user" not in self.messages[-1].values():
            self.messages.append({"role": "user", "content": prompt})
            self.messages.append({"role": "assistant", "content": llm_response})
        else:
            self.messages.append({"role": "assistant", "content": llm_response})

    # TODO: implementation is convoluted, find a more streamlined solution
    def prompt_llm(
            self,
            prompt: str,
            singleton: bool=False,
            sysprompt: str=None,
            extend_messages: bool=True) -> str:
        """
        Prompt the correct model based on hardware availability.
        params:
            singleton: if True, leverages get_singleton() for isolated llm interaction
            sysprompt: value will override default prompt for isolated interaction (get_singleton)
            extend_messages: if True, will append llm response to conversation 
            """
        # return only 1 interaction (system-user) for singletons and cpu hardware (limited context window)
        if singleton or self.device == "cpu":
            messages = self.get_singleton(prompt, sysprompt)
        else:
            messages = self.get_messages(prompt)
        response = self.model_selection[self.device][2](messages)
        if extend_messages:
            self.extend_conversation(prompt, response)
        return response

    def _prompt_gpu_llm(self, messages: list[dict]) -> str:
        """Prompt GPU model."""
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            padding=True).to(self.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.prompt_args)
        response = self.tokenizer.decode(outputs[0])
        # the response is a conversational list of dicts with roles and contents
        # we leverage the tags to extract the last response in the convo
        # print(response)
        # sometimes the chat template fails, thus we include a an optional pattern
        response = re.findall(
            #r"\>assistant\<\|end\_header\_id\|\>(.*?)\<\|eot\_id\|\>",
            # r"assistant(?:\<\|end\_header\_id\|\>)?\n\n(.*?)\<\|eot\_id\|\>",
            r"\n\n(.*?)\<\|eot\_id\|\>",
            response,
            re.DOTALL
            )[-1].strip()
        del inputs, outputs
        # don't wait for python garbage collection, flush instantenously
        gc.collect()
        torch.cuda.empty_cache()
        return response

    def _prompt_cpu_llm(self, messages: list[dict]) -> str:
        """Prompt CPU model."""
        outputs = self.model.create_chat_completion(
            messages=messages,
            max_tokens=200,
            **self.prompt_args
        )
        response = outputs["choices"][0]["message"]["content"]
        del outputs
        # don't wait for python garbage collection, flush instantenously
        gc.collect()
        return response

    def get_emb_model(self) -> SentenceTransformer:
        """Retreieve embedding model separatly (when not parsing documentation)"""
        emb_model = SentenceTransformer(str(self.emb_model_path)).to(self.device)
        return emb_model
