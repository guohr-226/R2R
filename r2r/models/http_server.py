import os
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"

import argparse
import json
import logging
import uvicorn
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import multiprocessing as mp

from sglang.srt.managers.io_struct import GenerateReqInput as SGLangGenerateReqInput

from r2r.models.sglang_patch.sl_disaggregation_system import SLDisaggregationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system: Optional[SLDisaggregationSystem] = None
server_args = None 

# Define request model
class GenerateReqInput(SGLangGenerateReqInput):
    # text: Optional[str] = None
    # input_ids: Optional[List[int]] = None
    # max_new_tokens: int = 2048
    # temperature: float = 0.0
    # top_p: float = 1.0
    # top_k: int = 100
    display_progress: bool = False
    return_trace: Optional[bool] = False


# OpenAI Chat Completion API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1
    return_trace: Optional[bool] = False
    trace_in_content: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    llm_ratio: float
    slm_token_count: Optional[int] = None
    llm_token_count: Optional[int] = None
    router_trigger_count: Optional[int] = None
    routed_token_count: Optional[int] = None
    token_trace: Optional[List[Dict[str, Any]]] = None
    endpoint_usage: Optional[UsageInfo] = None
    reference_usage: Optional[UsageInfo] = None
    dashscope_usage: Optional[UsageInfo] = None


def _result_value(result, key: str, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _usage_info_or_none(usage):
    if usage is None:
        return None
    if isinstance(usage, UsageInfo):
        return usage
    return UsageInfo(**usage)


def _usage_info_to_dict(usage: UsageInfo) -> Dict[str, int]:
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return usage.dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    print("Initializing R2R system inside lifespan...")

    if server_args:
        # Load config from path (file or folder)
        config_path = server_args.config_path
        with open(config_path, "r") as f:
            model_config = json.load(f)
        router_config = model_config.get("router", {})
        router_path = router_config.get("router_path")

        quick_tp_size_cfg = int(model_config.get("quick", {}).get("tp_size", 1))
        reference_tp_size_cfg = int(model_config.get("reference", {}).get("tp_size", 1))
        quick_cfg = model_config.get("quick", {})
        reference_cfg = model_config.get("reference", {})

        quick_tp_size_cli = getattr(server_args, "tp_size_quick", None)
        reference_tp_size_cli = getattr(server_args, "tp_size_ref", None)

        quick_tp_size = quick_tp_size_cfg if quick_tp_size_cli is None else int(quick_tp_size_cli)
        reference_tp_size = (
            reference_tp_size_cfg if reference_tp_size_cli is None else int(reference_tp_size_cli)
        )
        if quick_tp_size < 1 or reference_tp_size < 1:
            raise ValueError(
                f"tp_size must be >= 1, got quick={quick_tp_size}, reference={reference_tp_size}"
            )
        print(
            f"Using TP sizes: quick={quick_tp_size} "
            f"({'config' if quick_tp_size_cli is None else 'cli'}), "
            f"reference={reference_tp_size} "
            f"({'config' if reference_tp_size_cli is None else 'cli'})"
        )

        quick_sglang_kwargs = {
            "dtype": quick_cfg.get("dtype", "bfloat16"),
            "tp_size": quick_tp_size,
            "enable_return_hidden_states": True,
        }
        reference_sglang_kwargs = {
            "dtype": reference_cfg.get("dtype", "bfloat16"),
            "tp_size": reference_tp_size,
        }
        passthrough_keys = (
            "disable_cuda_graph",
            "cuda_graph_max_bs",
            "cuda_graph_bs",
            "disable_custom_all_reduce",
            "trust_remote_code",
            "max_prefill_tokens",
            "max_total_tokens",
            "kv_cache_dtype",
            "skip_server_warmup",
            "schedule_conservativeness",
            "mem_fraction_static",
        )
        for key in passthrough_keys:
            if key in quick_cfg:
                quick_sglang_kwargs[key] = quick_cfg[key]
            if key in reference_cfg:
                reference_sglang_kwargs[key] = reference_cfg[key]

        # Determine switching strategy first
        switching_strategy = router_config.get("switching_strategy")
        if switching_strategy is None:
            switching_strategy = "neural"
        print(f"Using switching strategy: {switching_strategy}")

        strategy_kwargs = {"model_path": router_path}

        # Threshold loading logic
        if switching_strategy == "neural":
            # Priority: config file's router.threshold > command line arg
            threshold = router_config.get("threshold")
            if threshold is None and server_args.threshold is not None:
                threshold = server_args.threshold
            
            if threshold is not None:
                strategy_kwargs["threshold"] = threshold
                print(f"Using neural threshold: {threshold}")
        else:
            # For non-neural strategies, use specific thresholds from config
            if "aleatoric_threshold" in router_config:
                strategy_kwargs["aleatoric_threshold"] = router_config["aleatoric_threshold"]
                print(f"Using aleatoric threshold from config: {router_config['aleatoric_threshold']}")
            
            if "entropy_threshold" in router_config:
                strategy_kwargs["entropy_threshold"] = router_config["entropy_threshold"]
                print(f"Using entropy threshold from config: {router_config['entropy_threshold']}")

        reference_backend = reference_cfg.get("backend", "sglang")

        try:
            if reference_backend == "dashscope_openai":
                from r2r.models.dashscope_hybrid_system import DashScopeHybridSystem

                system = DashScopeHybridSystem(
                    model_config=model_config,
                    switching_strategy=switching_strategy,
                    strategy_kwargs=strategy_kwargs,
                )
            else:
                system = SLDisaggregationSystem(
                    model_config=model_config,
                    device="cuda",
                    dtype="bfloat16",
                    switching_strategy=switching_strategy,
                    strategy_kwargs=strategy_kwargs,
                    quick_sglang_kwargs=quick_sglang_kwargs,
                    reference_sglang_kwargs=reference_sglang_kwargs,
                    overlap_tp_schedule=server_args.overlap_tp_schedule,
                    llm_min_batch_size=server_args.llm_min_batch_size,
                )
            print("System initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            raise e

    yield

    print("Shutting down system...")
    if system:
        # system.shutdown()
        pass

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_request(obj: GenerateReqInput):

    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        input_ids = obj.input_ids
        if input_ids is None:
            if obj.text is None:
                raise HTTPException(status_code=400, detail="Either text or input_ids must be provided")
            input_ids = system.tokenizer.encode(obj.text)

        default_sampling_params = {
            "max_new_tokens": 128,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        sampling_params = obj.sampling_params
        if sampling_params is None:
            sampling_params = default_sampling_params
        
        result = await system.generate_one_request(
            input_id=input_ids,
            max_new_tokens=sampling_params.get('max_new_tokens', 128),
            temperature=sampling_params.get('temperature', 1.0),
            top_p=sampling_params.get('top_p', 1.0),
            top_k=sampling_params.get('top_k', -1),
            display_progress=obj.display_progress
        )
        
        
        output_ids = _result_value(result, 'output_ids', [])
        output_text = _result_value(result, 'output_text')
        if output_text is None:
            output_text = system.tokenizer.decode(output_ids, skip_special_tokens=True)
        slm_token_count = _result_value(result, 'slm_token_count', 0)
        llm_token_count = _result_value(result, 'llm_token_count', 0)
        router_trigger_count = _result_value(result, 'router_trigger_count', llm_token_count)
        routed_token_count = _result_value(result, 'routed_token_count', llm_token_count)
        token_trace = _result_value(result, 'token_trace', [])
        reference_usage = _result_value(result, 'reference_usage', None)
        dashscope_usage = _result_value(result, 'dashscope_usage', reference_usage)
        endpoint_usage = {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
        }
        
        response = {
            "text": output_text,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "llm_ratio": _result_value(result, 'llm_ratio', None),
            "slm_token_count": slm_token_count,
            "llm_token_count": llm_token_count,
            "router_trigger_count": router_trigger_count,
            "routed_token_count": routed_token_count,
            "endpoint_usage": endpoint_usage,
            "reference_usage": reference_usage,
            "dashscope_usage": dashscope_usage,
        }
        if obj.display_progress or obj.return_trace:
            response["token_trace"] = token_trace
        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    
    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")
    
    try:
        # Convert messages to text using chat template
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        if hasattr(system, "generate_chat_completion"):
            result = await system.generate_chat_completion(
                messages=messages,
                max_new_tokens=request.max_tokens or 2048,
                temperature=request.temperature if request.temperature is not None else 1.0,
                top_p=request.top_p if request.top_p is not None else 1.0,
                top_k=-1,
            )
            input_ids = system.encode_messages(messages)
            output_ids = _result_value(result, "output_ids", [])
            output_text = _result_value(result, "output_text")
            if output_text is None:
                output_text = system.tokenizer.decode(output_ids, skip_special_tokens=True)
            prompt_tokens = len(input_ids)
            completion_tokens = len(output_ids)
            llm_ratio = _result_value(result, "llm_ratio", None)
        else:
            # Apply chat template if available
            if hasattr(system.tokenizer, 'apply_chat_template'):
                input_ids = system.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    tokenize=True
                )
            else:
                # Fallback: concatenate messages
                text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                input_ids = system.tokenizer.encode(text)
        
            prompt_tokens = len(input_ids)
            
            sampling_params = {
                "temperature": 1.0,
                "top_p":  1.0,
                "top_k": -1,
                "max_new_tokens": 2048,
            }

            if request.temperature is not None:
                sampling_params["temperature"] = request.temperature
            if request.top_p is not None:
                sampling_params["top_p"] = request.top_p
            if request.max_tokens is not None:
                sampling_params["max_new_tokens"] = request.max_tokens

            # Generate response
            result = await system.generate_one_request(
                input_id=input_ids,
                max_new_tokens=sampling_params["max_new_tokens"],
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                top_k=sampling_params["top_k"],
                display_progress=False
            )
            
            # Extract output
            output_ids = _result_value(result, 'output_ids', [])
            output_text = _result_value(result, 'output_text')
            if output_text is None:
                output_text = system.tokenizer.decode(output_ids, skip_special_tokens=True)
            completion_tokens = len(output_ids)
            llm_ratio = _result_value(result, 'llm_ratio', None)

        slm_token_count = _result_value(result, 'slm_token_count', 0)
        llm_token_count = _result_value(result, 'llm_token_count', 0)
        router_trigger_count = _result_value(result, 'router_trigger_count', llm_token_count)
        routed_token_count = _result_value(result, 'routed_token_count', llm_token_count)
        token_trace = _result_value(result, 'token_trace', [])
        reference_usage = _result_value(result, 'reference_usage', None)
        dashscope_usage = _result_value(result, 'dashscope_usage', reference_usage)
        endpoint_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        message_content = output_text
        if request.trace_in_content:
            message_content = json.dumps(
                {
                    "final_template": output_text,
                    "source": "hybrid" if router_trigger_count > 0 else "slm",
                    "router_trigger_count": router_trigger_count,
                    "routed_token_count": routed_token_count,
                    "endpoint_usage": _usage_info_to_dict(endpoint_usage),
                    "token_trace": token_trace,
                    "reference_usage": reference_usage,
                    "dashscope_usage": dashscope_usage,
                },
                ensure_ascii=False,
            )
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model or "default",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=message_content),
                    finish_reason="stop"
                )
            ],
            usage=endpoint_usage,
            llm_ratio=llm_ratio,
            slm_token_count=slm_token_count,
            llm_token_count=llm_token_count,
            router_trigger_count=router_trigger_count,
            routed_token_count=routed_token_count,
            token_trace=token_trace if request.return_trace else None,
            endpoint_usage=endpoint_usage,
            reference_usage=_usage_info_or_none(reference_usage),
            dashscope_usage=_usage_info_or_none(dashscope_usage),
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System initializing")
    return {"status": "healthy"}
