import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

import requests


DEFAULT_MODEL = "gpt-4.1-mini"
API_URL = "https://api.openai.com/v1/responses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a retail shelf image with a modern vision API."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-path", help="Absolute or relative path to a local image.")
    source.add_argument("--image-url", help="Public URL of an image to analyze.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Vision model to use (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def encode_image(path: str) -> str:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type for file: {file_path}")

    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_message_content(image_url: str) -> list[dict]:
    return [
        {
            "type": "input_text",
            "text": (
                "You are a computer vision assistant for an AI engineer portfolio. "
                "Inspect the retail shelf image and return strict JSON with the keys "
                "scene_summary, visible_products, stock_risks, misplacements, "
                "accessibility_notes, and recommended_actions."
            ),
        },
        {"type": "input_image", "image_url": image_url},
    ]


def build_input_from_path(image_path: str) -> list[dict]:
    local_image_data_url = encode_image(image_path)
    return [{"role": "user", "content": build_message_content(local_image_data_url)}]


def build_input_from_url(image_url: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": build_message_content(image_url),
        }
    ]


def extract_text(response_json: dict) -> str:
    output = response_json.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
            text = content.get("text")
            if isinstance(text, str) and text:
                return text
    raise ValueError("No text output found in API response.")


def analyze_image(input_payload: list[dict], model: str, api_key: str) -> str:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": input_payload,
        },
        timeout=60,
    )
    response.raise_for_status()
    return extract_text(response.json())


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required.")

    input_payload = (
        build_input_from_url(args.image_url)
        if args.image_url
        else build_input_from_path(args.image_path)
    )
    result = analyze_image(input_payload=input_payload, model=args.model, api_key=api_key)

    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
