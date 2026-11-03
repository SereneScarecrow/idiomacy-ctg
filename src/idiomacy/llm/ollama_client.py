"""
Ollama backend (local models).
"""

import json
import os
import time
import urllib.request

import ollama

from idiomacy.llm.base import LLMClient, LLMError, Request, Response


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, host: str | None = None, timeout: float | None = None):
        host = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self._host = host if "://" in host else f"http://{host}"
        self._client = ollama.Client(host=self._host, timeout=timeout)

    def generate(self, request: Request) -> Response:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        user_message = {"role": "user", "content": request.prompt}
        if request.images:
            user_message["images"] = list(request.images)
        messages.append(user_message)

        options = {"temperature": request.temperature}
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        options.update(request.extra)  # num_ctx, top_p, ...

        start = time.perf_counter()
        try:
            reply = self._client.chat(
                model=request.model,
                messages=messages,
                options=options,
                format=request.schema,
                think=request.reasoning,
                stream=False,
            )
        except ollama.ResponseError as error:
            raise LLMError(f"Ollama error {error.status_code}: {error.error}") from error
        except ConnectionError as error:
            raise LLMError(f"Cannot reach Ollama at {self._host}: {error}") from error
        latency = time.perf_counter() - start

        return Response(
            text=reply.message.content or "",
            finish_reason=reply.done_reason or "unknown",
            prompt_tokens=reply.prompt_eval_count,
            completion_tokens=reply.eval_count,
            latency_s=latency,
            model=reply.model or request.model,
            raw=reply.model_dump(mode="json"),
            reasoning=getattr(reply.message, "thinking", None),
        )

    def model_info(self, model: str) -> dict:
        try:
            installed = self._client.list().models
        except ConnectionError as error:
            raise LLMError(f"Cannot reach Ollama at {self._host}: {error}") from error
        for item in installed:
            if item.model in (model, f"{model}:latest"):
                return {
                    "model": item.model,
                    "digest": item.digest,
                    "details": item.details.model_dump(mode="json") if item.details else None,
                }
        raise LLMError(f"Model {model!r} is not installed (run: ollama pull {model})")

    def describe(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self._host}/api/version", timeout=5) as response:
                version = json.load(response)["version"]
        except OSError as error:
            raise LLMError(f"Cannot reach Ollama at {self._host}: {error}") from error
        return {"provider": self.provider, "host": self._host, "server_version": version}