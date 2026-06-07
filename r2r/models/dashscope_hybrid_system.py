import os
from typing import Dict, List, Optional

import torch
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.switching import create_switching_strategy


class DashScopeHybridSystem:
    """Hybrid system that uses local quick/router models and DashScope as reference.

    This mode is an API-backed approximation of R2R. The router is evaluated on the
    local quick model's next-token state; when it selects the reference model, the
    full request is delegated to DashScope's OpenAI-compatible chat API.
    """

    def __init__(
        self,
        model_config: Dict,
        switching_strategy: str = "neural",
        strategy_kwargs: Optional[dict] = None,
    ):
        self.model_config = model_config
        self.quick_config = model_config["quick"]
        self.reference_config = model_config["reference"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        quick_model_path = self.quick_config["model_path"]
        print(f"Loading quick model {quick_model_path} for DashScope hybrid mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            quick_model_path,
            trust_remote_code=self.quick_config.get("trust_remote_code", True),
        )
        self.quick_model = AutoModelForCausalLM.from_pretrained(
            quick_model_path,
            torch_dtype=self.dtype,
            trust_remote_code=self.quick_config.get("trust_remote_code", True),
        ).to(self.device)
        self.quick_model.eval()

        router_config = model_config.get("router", {})
        merged_strategy_kwargs = {**(strategy_kwargs or {})}
        override_init_args = router_config.get("override_init_args", {})
        merged_strategy_kwargs["override_init_args"] = override_init_args
        merged_strategy_kwargs["device"] = self.device
        merged_strategy_kwargs["use_cuda_graph"] = False
        self.switching_strategy = create_switching_strategy(
            switching_strategy,
            **merged_strategy_kwargs,
        )

        base_url = self._resolve_required_value("base_url", "base_url_env")
        api_key = self._resolve_required_value("api_key", "api_key_env")
        self.reference_model_name = self.reference_config["model_name"]
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _resolve_required_value(self, value_key: str, env_key: str) -> str:
        value = self.reference_config.get(value_key)
        if value:
            return value

        env_name = self.reference_config.get(env_key)
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
            raise RuntimeError(f"Environment variable {env_name} is required")

        raise RuntimeError(f"reference.{value_key} or reference.{env_key} is required")

    @torch.no_grad()
    def _route_input(self, input_id: List[int]) -> str:
        input_tensor = torch.tensor([input_id], device=self.device, dtype=torch.long)
        outputs = self.quick_model(
            input_ids=input_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = outputs.logits[:, -1:, :]
        hidden_states = [outputs.hidden_states[-1][:, -1:, :]]
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        model_outputs = ModelOutputs(
            logits=logits,
            hidden_states=hidden_states,
            token=next_token,
        )
        choice = self.switching_strategy.route(model_outputs)
        return "reference" if choice.any().item() else "quick"

    async def _generate_dashscope(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.reference_model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content or ""

    def encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        try:
            if getattr(self.tokenizer, "chat_template", None):
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
        except Exception as exc:
            print(f"Failed to apply chat template, falling back to plain text: {exc}")

        text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        text += "\nassistant:"
        return self.tokenizer.encode(text)

    @torch.no_grad()
    def _generate_quick(
        self,
        input_id: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[int]:
        input_tensor = torch.tensor([input_id], device=self.device, dtype=torch.long)
        do_sample = temperature is not None and temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = top_k
        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }
        output = self.quick_model.generate(input_tensor, **generation_kwargs)
        return output[0, input_tensor.shape[-1] :].detach().cpu().tolist()

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> Dict:
        input_id = self.encode_messages(messages)
        selected_model = self._route_input(input_id)
        if selected_model == "reference":
            output_text = await self._generate_dashscope(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        else:
            output_ids = self._generate_quick(
                input_id=input_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return {
            "output_ids": output_ids,
            "output_text": output_text,
            "source_model": selected_model,
            "llm_ratio": 1.0 if selected_model == "reference" else 0.0,
        }

    async def generate_one_request(
        self,
        input_id: List[int],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        display_progress: bool = False,
    ) -> Dict:
        selected_model = self._route_input(input_id)
        if selected_model == "reference":
            prompt_text = self.tokenizer.decode(input_id, skip_special_tokens=False)
            output_text = await self._generate_dashscope(
                messages=[{"role": "user", "content": prompt_text}],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        else:
            output_ids = self._generate_quick(
                input_id=input_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return {
            "output_ids": output_ids,
            "output_text": output_text,
            "source_model": selected_model,
            "llm_ratio": 1.0 if selected_model == "reference" else 0.0,
        }
