"""Bottega — a small MCP server that lets Claude commission work from other
models on OpenRouter: write the prompt, pick the model, bring back the result.

Tools
  ask(model, prompt, system, image_urls)            -> text (and images, if the model returns any)
  paint(prompt, model, aspect_ratio, resolution, reference_urls) -> one image
  models(kind, search)                               -> model slugs with prices

Env
  OPENROUTER_API_KEY   required
  PAINT_MODEL          default image model slug
  ASK_MODEL            default text model slug
  SPEND_CAP_USD        soft daily cap, counted in this process (default 2)
  BOTTEGA_PATH         secret URL path, e.g. /mcp-<random>; default /mcp
  PORT                 set by Render
"""

import base64
import datetime as dt
import os

import httpx
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings

API = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY", "")
PAINT_MODEL = os.environ.get("PAINT_MODEL", "google/gemini-3.1-flash-image-preview")
ASK_MODEL = os.environ.get("ASK_MODEL", "openai/gpt-5-mini")
CAP = float(os.environ.get("SPEND_CAP_USD", "2"))

_spent = {"day": None, "usd": 0.0}

mcp = FastMCP(
    "bottega",
    instructions=(
        "Commission work from other models on OpenRouter. You write the prompt; "
        "they produce. Use when a task needs a capability you lack (images, a "
        "specialist model, a second opinion). Call models() if a slug is rejected. "
        "Every reply ends with a cost line; there is a soft daily spend cap."
    ),
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "8000")),
    streamable_http_path=os.environ.get("BOTTEGA_PATH", "/mcp"),
    stateless_http=True,
    # Render sits behind a proxy with its own hostname; the default
    # DNS-rebinding guard only allows localhost and would reject every call.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _headers() -> dict:
    if not KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {"Authorization": f"Bearer {KEY}", "X-Title": "Bottega"}


def _check_cap() -> str | None:
    today = dt.date.today().isoformat()
    if _spent["day"] != today:
        _spent.update(day=today, usd=0.0)
    if _spent["usd"] >= CAP:
        return f"Daily cap reached: ${_spent['usd']:.3f} of ${CAP:.2f}. Ask Qi to raise SPEND_CAP_USD."
    return None


def _book(usage: dict | None, model: str) -> str:
    cost = (usage or {}).get("cost")
    if isinstance(cost, (int, float)):
        _spent["usd"] += cost
    return f"model={model} cost={cost} today=${_spent['usd']:.3f}/${CAP:.2f}"


def _image_from_data_url(url: str) -> Image | None:
    if not url.startswith("data:") or ";base64," not in url:
        return None
    head, b64 = url.split(";base64,", 1)
    fmt = head.split("/")[-1] or "png"
    return Image(data=base64.b64decode(b64), format=fmt)


@mcp.tool()
async def ask(model: str = "", prompt: str = "", system: str = "", image_urls: list[str] | None = None,
              max_tokens: int = 4000) -> list:
    """Send a prompt to any chat model on OpenRouter and return its answer.

    model: slug like 'openai/gpt-5-mini'; empty = ASK_MODEL. See models('text').
    system: optional system prompt.
    image_urls: optional https or data: URLs the model should look at (vision models only).
    Models that can output images return them too.
    """
    if (msg := _check_cap()):
        return [msg]
    model = model or ASK_MODEL
    content: list | str = prompt
    if image_urls:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in image_urls
        ]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{API}/chat/completions", headers=_headers(), json=body)
    if r.status_code >= 400:
        return [f"OpenRouter {r.status_code}: {r.text[:800]}"]
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    out: list = [message.get("content") or "(no text)"]
    for img in message.get("images") or []:
        pic = _image_from_data_url(((img or {}).get("image_url") or {}).get("url", ""))
        if pic:
            out.append(pic)
    out.append(_book(data.get("usage"), data.get("model", model)))
    return out


@mcp.tool()
async def paint(prompt: str, model: str = "", aspect_ratio: str = "3:4", resolution: str = "",
                reference_urls: list[str] | None = None) -> list:
    """Generate one image from a text prompt and return it.

    model: image model slug; empty = PAINT_MODEL. See models('image').
    aspect_ratio: e.g. 1:1, 3:4, 4:3, 9:16, 16:9, or 'auto'.
    resolution: optional tier, '1K' / '2K' / '4K'.
    reference_urls: optional input images for image-to-image (https or data: URLs).
    """
    if (msg := _check_cap()):
        return [msg]
    model = model or PAINT_MODEL
    body: dict = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio}
    if resolution:
        body["resolution"] = resolution
    if reference_urls:
        body["input_references"] = reference_urls
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{API}/images", headers=_headers(), json=body)
    if r.status_code >= 400:
        return [f"OpenRouter {r.status_code}: {r.text[:800]}"]
    data = r.json()
    item = (data.get("data") or [{}])[0]
    b64 = item.get("b64_json")
    if not b64:
        return [f"No image in response: {str(data)[:800]}"]
    fmt = (item.get("media_type") or "image/png").split("/")[-1].replace("svg+xml", "svg")
    return [Image(data=base64.b64decode(b64), format=fmt), _book(data.get("usage"), model)]


@mcp.tool()
async def models(kind: str = "text", search: str = "", limit: int = 40) -> str:
    """List model slugs on OpenRouter with prices.

    kind: 'text' (chat models) or 'image' (image generators).
    search: optional substring filter on slug or name, e.g. 'claude', 'flux', 'vision'.
    """
    url = f"{API}/images/models" if kind == "image" else f"{API}/models"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers=_headers())
    if r.status_code >= 400:
        return f"OpenRouter {r.status_code}: {r.text[:800]}"
    rows = []
    for m in r.json().get("data", []):
        slug, name = m.get("id", "?"), m.get("name", "")
        if search and search.lower() not in f"{slug} {name}".lower():
            continue
        p = m.get("pricing") or {}
        mods = ",".join((m.get("architecture") or {}).get("input_modalities") or [])
        price = f"in {p.get('prompt', '?')} / out {p.get('completion', '?')} per token" if p else ""
        rows.append(" | ".join(x for x in (slug, mods and f"input:{mods}", price) if x))
        if len(rows) >= limit:
            break
    return "\n".join(rows) or "(no match)"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
