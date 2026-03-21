# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import json
import os
import urllib.request
from typing import Any, Dict, List

from litellm import completion


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])
        self.extra_params = config.get("EXTRA_PARAMS", {}) or {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        # 添加 API Key
        if self.api_key:
            params["api_key"] = self.api_key

        # 添加 API Base（如果配置了）
        if self.api_base:
            params["api_base"] = self.api_base

        # 添加 max_tokens（如果配置了且不为 0）
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        # 添加 fallback 模型（如果配置了）
        if self.fallback_models:
            params["fallbacks"] = self.fallback_models

        # 部分自定义 OpenAI 兼容网关会拦截默认 Python User-Agent。
        # 为避免 GitHub Actions / Python 客户端被误判，给自定义端点设置一个中性 UA。
        if self.api_base and "extra_headers" not in self.extra_params:
            params["extra_headers"] = {"User-Agent": "curl/8.7.1"}

        # 合并配置里的额外参数（如 extra_headers、top_p、stop 等）
        for key, value in self.extra_params.items():
            if key == "extra_headers" and isinstance(value, dict):
                merged_headers = dict(params.get("extra_headers", {}))
                merged_headers.update(value)
                params["extra_headers"] = merged_headers
            elif key not in params:
                params[key] = value

        # 合并其他额外参数
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 调用 LiteLLM。部分自定义网关会错误返回 SSE 流，
        # 导致 LiteLLM 按普通 JSON 解析时报 invalid response。
        try:
            response = completion(**params)
        except Exception as e:
            if self.api_base and self._should_fallback_to_raw_http(str(e)):
                return self._chat_via_raw_http(messages, **kwargs)
            raise

        # 提取响应内容
        # 某些模型/提供商返回 list（内容块）而非 str，统一转为 str
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def _should_fallback_to_raw_http(self, error_text: str) -> bool:
        markers = [
            "Empty or invalid response from LLM endpoint",
            "chat.completion.chunk",
            "data: {",
            "text/event-stream",
        ]
        return any(marker in error_text for marker in markers)

    def _chat_via_raw_http(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        base = self.api_base.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

        payload = {
            "model": self.model.split("/", 1)[1] if "/" in self.model else self.model,
            "messages": messages,
            "stream": False,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "curl/8.7.1",
        }
        extra_headers = self.extra_params.get("extra_headers", {})
        if isinstance(extra_headers, dict):
            headers.update(extra_headers)

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )

        with urllib.request.urlopen(request, timeout=kwargs.get("timeout", self.timeout)) as response:
            body = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get_content_type()

        if content_type == "text/event-stream" or body.startswith("data: "):
            content = self._parse_sse_content(body)
            if content:
                return content
            raise ValueError("AI 网关返回了 SSE 流，但其中没有可提取的文本内容")

        data = json.loads(body)
        return self._extract_openai_content(data)

    def _parse_sse_content(self, body: str) -> str:
        parts: List[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            chunk = line[6:].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            content = self._extract_openai_content(data)
            if content:
                parts.append(content)
                continue

            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                delta_content = delta.get("content")
                if isinstance(delta_content, str) and delta_content:
                    parts.append(delta_content)

        return "".join(parts).strip()

    def _extract_openai_content(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        if isinstance(content, str):
            return content
        return ""

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.api_key:
            return False, "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
