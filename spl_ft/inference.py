"""Small OpenAI Chat Completions endpoint; no vLLM/ARM kernel dependency."""
import copy
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


def decode_output(text, tools=None, thinking_prefix=True):
    reasoning = None
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    elif thinking_prefix:
        # A truncated reasoning-only generation must not become a fake final answer.
        return {"role": "assistant", "content": "", "reasoning_content": text.removeprefix("<think>")}
    schemas = {t.get("function", t)["name"]: t.get("function", t) for t in tools or []}
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        fn = re.search(r"<function=([^>]+)>(.*?)</function>", block, re.S)
        if not fn:
            continue
        name, body = fn.groups()
        properties = schemas.get(name, {}).get("parameters", {}).get("properties", {})
        args = {}
        for key, value in re.findall(r"<parameter=([^>]+)>\s*\n?(.*?)\n?</parameter>", body, re.S):
            kind = properties.get(key, {}).get("type", "string")
            value = value.strip("\n")
            if kind != "string":
                value = json.loads(value)
            args[key] = value
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    result = {"role": "assistant", "content": content or (None if calls else "")}
    if reasoning is not None:
        result["reasoning_content"] = reasoning
    if calls:
        result["tool_calls"] = calls
    return result


def normalize_messages(messages):
    result = copy.deepcopy(messages)
    for m in result:
        for call in m.get("tool_calls", []):
            fn = call.get("function", call)
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    return result


def create_inference_app(model_path, revision, context_length, defaults):
    state = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path
        source = model_path
        adapter = Path(source) / "adapter_config.json"
        base = "LiquidAI/LFM3-1.7B" if adapter.exists() else source
        remote = not Path(base).exists()
        kw = {"revision": revision} if remote else {}
        state["tokenizer"] = AutoTokenizer.from_pretrained(base, trust_remote_code=True, **kw)
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", **kw).to("cuda")
        if adapter.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, source).merge_and_unload()
        model.eval()
        state["model"] = model
        yield
        del state["model"]
        torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if request.headers.get("authorization") != "Bearer " + os.environ["SPL_INFER_KEY"]:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "lfm3", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def complete(body: dict):
        import torch
        tokenizer = state["tokenizer"]
        tools = body.get("tools") or None
        try:
            messages = normalize_messages(body["messages"])
            inputs = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", preserve_thinking=True)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid chat request: {e}") from e
        prompt_tokens = inputs["input_ids"].shape[-1]
        max_new = min(int(body.get("max_completion_tokens") or body.get("max_tokens") or defaults["max_new_tokens"]), defaults["max_new_tokens"])
        if prompt_tokens + max_new > context_length:
            raise HTTPException(400, {"type": "context_length_exceeded", "message": f"{prompt_tokens}+{max_new} exceeds {context_length}"})
        temperature = float(body.get("temperature", defaults["temperature"]))
        generation = {"max_new_tokens": max_new, "do_sample": temperature > 0,
                      "repetition_penalty": 1.1, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        if temperature > 0:
            generation.update(temperature=temperature, top_k=50)
        stops = body.get("stop")
        if stops:
            generation.update(stop_strings=[stops] if isinstance(stops, str) else stops, tokenizer=tokenizer)
        with lock, torch.inference_mode():
            # Seed once on startup; don't reset the RNG on every agent turn.
            output = state["model"].generate(**inputs.to("cuda"), **generation)
        ids = output[0, prompt_tokens:]
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        message = decode_output(raw, tools)
        finish = "tool_calls" if message.get("tool_calls") else "length" if len(ids) >= max_new else "stop"
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(ids), "total_tokens": prompt_tokens + len(ids)}
        common = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": "lfm3"}
        if not body.get("stream"):
            return {**common, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}

        def events():
            delta = {k: v for k, v in message.items() if v is not None}
            if "tool_calls" in delta:
                delta["tool_calls"] = [{"index": i, **c} for i, c in enumerate(delta["tool_calls"])]
            for change, reason in ((delta, None), ({}, finish)):
                yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": change, "finish_reason": reason}]}) + "\n\n"
            yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    return app


def main():
    import argparse
    import uvicorn
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    c = json.load(open(args.config))
    set_seed(c["seed"])
    uvicorn.run(create_inference_app(c["model_path"], c["revision"], c["context_length"], c), host="0.0.0.0", port=c["port"], log_level="warning")


if __name__ == "__main__":
    main()
