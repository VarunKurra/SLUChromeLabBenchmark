# This file wires up a bunch of model providers so the runner can call them the same way.
# Each one has a small wrapper that takes a prompt and an image path and returns a string.

# Image handling is a little different depending on the provider:
# - "pil" -> a PIL image for Google GenAI
# - "url" -> a hosted image URL for Groq, NVIDIA, and HCompany
# - "base64" -> a data URL for HuggingFace and OpenRouter

# If you want to turn a model off for a bit, set "enabled" to False.

import os
import base64
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Small compatibility fix for newer google-genai versions where cleanup can trip over the async client.
def _patch_google_genai_aclose() -> None:
    try:
        from google.genai import _api_client as google_api_client
    except Exception:
        return

    if getattr(google_api_client.BaseApiClient, "_slu_patched", False):
        return

    async def patched_aclose(self) -> None:
        try:
            async_client = getattr(self, "_async_httpx_client", None)
            if async_client is not None:
                await async_client.aclose()
        except Exception:
            pass

        aiohttp_sessions = getattr(self, "_aiohttp_sessions", None)
        if not aiohttp_sessions:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for loop, session in list(aiohttp_sessions.items()):
            if loop == current_loop:
                try:
                    await session.close()
                except Exception:
                    pass
            elif loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(session.close(), loop)
                except Exception:
                    pass

    google_api_client.BaseApiClient.aclose = patched_aclose
    google_api_client.BaseApiClient._slu_patched = True


_patch_google_genai_aclose()

# Shared image URL

IMAGE_URL = "https://www.image2url.com/r2/default/images/1780602309583-a0c99451-72f0-4294-babb-7bc69cb09002.webp"

# Small helpers

# Turn an image file into a base64 string so some providers can use it directly.
def _encode_base64(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "image/webp" if ext == "webp" else f"image/{ext}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# Pick an image URL or fall back to a default one if the file is missing.
def _image_url(image_path: Optional[str]) -> str:
    if image_path and Path(image_path).exists():
        return _encode_base64(image_path)
    return IMAGE_URL


# Build the prompt that gets sent to the model, including the CSV data when it exists.
def _build_prompt(prompt: str, csv_path: Optional[str]) -> str:
    base_prompt = (prompt or "").strip()
    instruction = (
        "You are answering a benchmark question from a chart or CSV. "
        "Return ONLY the final answer. "
        "Do not include reasoning, explanation, markdown, bullets, labels, punctuation, or extra words. "
        "If the question asks for a number, output only the number with exactly two decimal places. "
        "If the question asks for a word or phrase, output only that word or phrase. "
        "If the question asks for a category, output only the category name. "
        "If the answer cannot be determined from the provided image or CSV, output exactly: insufficient information"
    )

    if not base_prompt:
        final_prompt = instruction
    else:
        final_prompt = f"{base_prompt}\n\n{instruction}"

    if not csv_path:
        return final_prompt

    path = Path(csv_path)
    if not path.exists():
        return final_prompt

    csv_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not csv_text:
        return final_prompt

    return f"{final_prompt}\n\nGraph CSV data:\n{csv_text}"


def _pil_image(image_path: str):
    from PIL import Image
    return Image.open(image_path)

# Google GenAI uses a PIL image.

def _make_google_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        import os as _os
        _os.environ.setdefault("GRPC_VERBOSITY", "NONE")
        _os.environ.setdefault("GLOG_minloglevel", "3")
        from google import genai as _genai
        from google.genai import types as _types
        client = _genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        image = _pil_image(image_path)
        loop = asyncio.get_event_loop()

        config_kwargs = {
            "temperature": 0.1,
            "top_p": 1.0,
            "max_output_tokens": 32,
            "system_instruction": (
                "You are answering a benchmark question from a chart or CSV. "
                "Follow the user's instruction exactly. "
                "Return only the final answer. "
                "Do not include reasoning, explanation, markdown, bullets, labels, punctuation, or extra words."
            ),
        }
        is_gemma_model = model_id.lower().startswith(("gemma-"))
        if is_gemma_model:
            try:
                config_kwargs["thinking_config"] = _types.ThinkingConfig(
                    thinkingLevel=_types.ThinkingLevel.MINIMAL,
                )
            except Exception:
                try:
                    config_kwargs["thinking_config"] = _types.ThinkingConfig(
                        includeThoughts=False,
                        thinkingBudget=0,
                    )
                except Exception:
                    try:
                        config_kwargs["thinking_config"] = _types.ThinkingConfig(
                            thinking_level="minimal"
                        )
                    except Exception:
                        pass

        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_id,
                    contents=[image, _build_prompt(prompt, csv_path)],
                    config=_types.GenerateContentConfig(**config_kwargs)
                )
            )
        except Exception as exc:
            if "thinking" in str(exc).lower() and "thinking_config" in config_kwargs:
                config_kwargs.pop("thinking_config", None)
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_id,
                        contents=[image, _build_prompt(prompt, csv_path)],
                        config=_types.GenerateContentConfig(**config_kwargs)
                    )
                )
            else:
                raise
        parts = getattr(getattr(response.candidates[0].content, "parts", []), "__iter__", lambda: iter([]))()
        text_chunks = []
        for part in parts:
            text_value = getattr(part, "text", None)
            if text_value:
                text_chunks.append(text_value)
        return "".join(text_chunks).strip()
    call.__name__ = model_id
    return call

# Groq uses an image URL.

def _make_groq_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(prompt, csv_path)},
                        {"type": "image_url", "image_url": {"url": _image_url(image_path)}}
                    ]
                }],
                temperature=0.0,
                max_tokens=256,
            )
        )
        return response.choices[0].message.content
    call.__name__ = model_id
    return call

# HuggingFace uses base64 data for the image.

def _hf_provider(model_id: str) -> str:
    m = model_id.lower()
    if "qwen3-vl-235b" in m or "glm-4.6v" in m:
        return "novita"
    if "kimi" in m:
        return "fireworks-ai"
    return "featherless-ai"

def _hf_extra_body(model_id: str) -> dict:
    m = model_id.lower()
    if any(x in m for x in ["qwen", "reasoning", "phi"]):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}

def _make_hf_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        from huggingface_hub import InferenceClient
        provider = _hf_provider(model_id)
        extra = _hf_extra_body(model_id)
        client = InferenceClient(provider=provider, api_key=os.getenv("HF_TOKEN"))
        b64_url = _encode_base64(image_path)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(prompt, csv_path)},
                        {"type": "image_url", "image_url": {"url": b64_url}}
                    ]
                }],
                max_tokens=256,
                extra_body={**extra, "temperature": 0.0} if extra else {"temperature": 0.0}
            )
        )
        return response.choices[0].message.content
    call.__name__ = model_id
    return call

# NVIDIA uses the OpenAI-style client with an image URL.

def _make_nvidia_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(prompt, csv_path)},
                        {"type": "image_url", "image_url": {"url": _image_url(image_path)}}
                    ]
                }],
                temperature=0.0,
                max_tokens=256
            )
        )
        return response.choices[0].message.content
    call.__name__ = model_id
    return call

# OpenRouter also uses the OpenAI-style client, but with a base64 image.

def _make_openrouter_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        b64_url = _encode_base64(image_path)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(prompt, csv_path)},
                        {"type": "image_url", "image_url": {"url": b64_url}}
                    ]
                }],
                temperature=0.0,
                max_tokens=256,
            )
        )
        return response.choices[0].message.content
    call.__name__ = model_id
    return call

# HCompany uses the same kind of setup as the others, just with a different endpoint.

def _make_hcompany_caller(model_id: str):
    async def call(prompt: str, image_path: str, csv_path: Optional[str] = None) -> str:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.hcompany.ai/v1/",
            api_key=os.getenv("HCOMPANY_API_KEY")
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(prompt, csv_path)},
                        {"type": "image_url", "image_url": {"url": _image_url(image_path)}}
                    ]
                }],
                temperature=0.0,
                max_tokens=256,
            )
        )
        return response.choices[0].message.content
    call.__name__ = model_id
    return call

# The list of models the runner can use.

MODELS = [
    # File 1: Google GenAI
    # Disabled 'gemini-3.5-flash'
    {"id": "gemini-3.5-flash",      "arch": "google", "enabled": False,  "call": _make_google_caller("gemini-3.5-flash")},

    {"id": "gemini-3.1-flash-lite",      "arch": "google", "enabled": True,  "call": _make_google_caller("gemini-3.1-flash-lite")},
    {"id": "gemini-2.5-flash",      "arch": "google", "enabled": False,  "call": _make_google_caller("gemini-2.5-flash")},
    {"id": "gemma-4-31b-it",        "arch": "google", "enabled": True,  "call": _make_google_caller("gemma-4-31b-it")},
    {"id": "gemma-4-26b-a4b-it",    "arch": "google", "enabled": True,  "call": _make_google_caller("gemma-4-26b-a4b-it")},

    # File 2: Groq
    {"id": "meta-llama/llama-4-scout-17b-16e-instruct",    "arch": "groq", "enabled": True,  "call": _make_groq_caller("meta-llama/llama-4-scout-17b-16e-instruct")},
    {"id": "qwen/qwen3.6-27b",                              "arch": "groq", "enabled": True,  "call": _make_groq_caller("qwen/qwen3.6-27b")},
    
    # File 3: HuggingFace InferenceClient
    # Disabled HuggingFace Models
    {"id": "moonshotai/Kimi-K2.6",                                         "arch": "huggingface", "enabled": False, "call": _make_hf_caller("moonshotai/Kimi-K2.6")},
    {"id": "Qwen/Qwen3-VL-235B-A22B-Instruct",                             "arch": "huggingface", "enabled": False, "call": _make_hf_caller("Qwen/Qwen3-VL-235B-A22B-Instruct")},
    {"id": "Qwen/Qwen3.6-27B",                                             "arch": "huggingface", "enabled": False, "call": _make_hf_caller("Qwen/Qwen3.6-27B")},
    {"id": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",    "arch": "huggingface", "enabled": False, "call": _make_hf_caller("Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")},
    {"id": "DavidAU/Qwen3.5-27B-Claude-4.6-OS-INSTRUCT",                  "arch": "huggingface", "enabled": False, "call": _make_hf_caller("DavidAU/Qwen3.5-27B-Claude-4.6-OS-INSTRUCT")},

    # File 4: NVIDIA
    {"id": "microsoft/phi-4-multimodal-instruct",           "arch": "nvidia", "enabled": False, "call": _make_nvidia_caller("microsoft/phi-4-multimodal-instruct")},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "arch": "nvidia", "enabled": True,  "call": _make_nvidia_caller("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")},
    {"id": "nvidia/nemotron-nano-12b-v2-vl",                "arch": "nvidia", "enabled": True,  "call": _make_nvidia_caller("nvidia/nemotron-nano-12b-v2-vl")},
    {"id": "mistralai/mistral-large-3-675b-instruct-2512",  "arch": "nvidia", "enabled": True,  "call": _make_nvidia_caller("mistralai/mistral-large-3-675b-instruct-2512")},
    {"id": "mistralai/ministral-14b-instruct-2512",         "arch": "nvidia", "enabled": True,  "call": _make_nvidia_caller("mistralai/ministral-14b-instruct-2512")},
    {"id": "meta/llama-4-maverick-17b-128e-instruct",       "arch": "nvidia", "enabled": True,  "call": _make_nvidia_caller("meta/llama-4-maverick-17b-128e-instruct")},
    
    # File 5: OpenRouter
    # Disabled 'nex-agi/nex-n2-pro:free'
    {"id": "nex-agi/nex-n2-pro:free", "arch": "openrouter", "enabled": False, "call": _make_openrouter_caller("nex-agi/nex-n2-pro:free")},
    
    # File 6: HCompany
    {"id": "holo3-1-35b-a3b", "arch": "hcompany", "enabled": True, "call": _make_hcompany_caller("holo3-1-35b-a3b")},
]

# Return only the models that are currently turned on.
def get_active_models() -> list:
    return [m for m in MODELS if m.get("enabled", True)]