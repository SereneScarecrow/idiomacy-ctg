"""
GigaChat backend (Sber API), used for the LLM judge.

Notes:
- The key is read from GIGACHAT_API_KEY (.env), the name from the README; the SDK's own
  variable GIGACHAT_CREDENTIALS also works. TLS, scope and timeout can be set in the
  constructor or through the SDK variables GIGACHAT_CA_BUNDLE_FILE, GIGACHAT_VERIFY_SSL_CERTS,
  GIGACHAT_SCOPE, GIGACHAT_TIMEOUT.
- The SDK call used is client.chat(Chat(...)): it exists in gigachat 0.1.x and is kept
  in 0.2.x as a (non-deprecated) compatibility method, so the code works with both.
- GigaChat has no `seed` option: request.seed is ignored. `reasoning` and `images` are
  not supported either (images raise LLMError).
- The model name is an alias that moves to new versions; the exact version that answered
  (e.g. 'GigaChat-2-Max:2.0.30.01') is returned by the API and saved in Response.model.
"""

import importlib.metadata
import os
import time

import httpx
from gigachat import GigaChat
from gigachat.exceptions import GigaChatException
from gigachat.models import Chat, Messages, MessagesRole

from idiomaticity.llm.base import LLMClient, LLMError, Request, Response

from dotenv import load_dotenv


class GigaChatClient(LLMClient):
    provider = "gigachat"

    def __init__(
        self,
        credentials: str | None = None,
        scope: str | None = None,
        ca_bundle_file: str | None = None,
        verify_ssl_certs: bool | None = None,
        timeout: float | None = None,
    ):
        if load_dotenv is not None:
            load_dotenv()
        credentials = (
            credentials
            or os.environ.get("GIGACHAT_API_KEY")
            or os.environ.get("GIGACHAT_CREDENTIALS")
        )
        if not credentials:
            raise LLMError("No GigaChat key: set GIGACHAT_API_KEY in .env")

        options = {
            "credentials": credentials,
            "scope": scope,
            "ca_bundle_file": ca_bundle_file,
            "verify_ssl_certs": verify_ssl_certs,
            "timeout": timeout,
        }
        # only what is set: the rest comes from the SDK defaults / GIGACHAT_* variables
        self._client = GigaChat(**{key: value for key, value in options.items() if value is not None})
        self._scope = scope or os.environ.get("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"

    def generate(self, request: Request) -> Response:
        if request.images:
            raise LLMError("GigaChatClient does not support images yet")

        messages = []
        if request.system:
            messages.append(Messages(role=MessagesRole.SYSTEM, content=request.system))
        messages.append(Messages(role=MessagesRole.USER, content=request.prompt))

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.schema is not None:
            payload["response_format"] = {"type": "json_schema", "schema": request.schema, "strict": True}
        payload.update(request.extra)  # top_p, repetition_penalty, ...

        start = time.perf_counter()
        try:
            reply = self._client.chat(Chat(**payload))
        except (GigaChatException, httpx.HTTPError) as error:
            raise LLMError(f"GigaChat error: {error}") from error
        latency = time.perf_counter() - start

        if not reply.choices:
            raise LLMError("GigaChat returned no choices")
        choice = reply.choices[0]
        usage = reply.usage
        raw = reply.model_dump(mode="json") if hasattr(reply, "model_dump") else reply.dict()

        return Response(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_s=latency,
            model=reply.model or request.model,
            raw=raw,
        )

    def model_info(self, model: str) -> dict:
        try:
            available = self._client.get_models().data
        except (GigaChatException, httpx.HTTPError) as error:
            raise LLMError(f"Cannot get the model list from GigaChat: {error}") from error
        for item in available:
            if item.id_ == model:
                return {"model": item.id_, "owned_by": item.owned_by}
        names = sorted(item.id_ for item in available)
        raise LLMError(f"Model {model!r} is not available in GigaChat (available: {names})")

    def describe(self) -> dict:
        return {
            "provider": self.provider,
            "scope": self._scope,
            "sdk_version": importlib.metadata.version("gigachat"),
        }