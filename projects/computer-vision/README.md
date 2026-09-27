# Vision-Based Retail Shelf Auditor

This portfolio project shows how to build a modern computer vision workflow using the latest OpenAI Responses API with image input.

## Problem

Retail teams need a fast way to review shelf photos and detect:

- out-of-stock areas
- misplaced products
- poor shelf presentation
- accessibility-friendly summaries for operations teams

## Solution

The CLI in `src/vision_portfolio.py` accepts either:

- a local image path, or
- a public image URL

It sends the image to a current multimodal vision model and asks for a structured JSON assessment with:

- `scene_summary`
- `visible_products`
- `stock_risks`
- `misplacements`
- `accessibility_notes`
- `recommended_actions`

## Why this is portfolio-worthy

This project demonstrates practical AI engineering patterns:

- multimodal API integration
- safe secret management with environment variables
- converting vision outputs into structured data
- clear documentation for reproducibility
- a business-facing computer vision use case

## Setup

```bash
cd /home/runner/work/AI-Engineer-Portfolio/AI-Engineer-Portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r projects/computer-vision/requirements.txt
export OPENAI_API_KEY="your_api_key"
```

## Usage

Analyze a local image:

```bash
python projects/computer-vision/src/vision_portfolio.py \
  --image-path /absolute/path/to/shelf.jpg
```

Analyze a public image URL:

```bash
python projects/computer-vision/src/vision_portfolio.py \
  --image-url "https://example.com/shelf.jpg"
```

Use a different model if needed:

```bash
python projects/computer-vision/src/vision_portfolio.py \
  --image-url "https://example.com/shelf.jpg" \
  --model gpt-4.1-mini
```

## Output

The script prints formatted JSON so the result can be:

- reviewed by a hiring manager
- stored in a pipeline
- sent to dashboards or automation tools

## Files

- `src/vision_portfolio.py` — main portfolio project code
- `requirements.txt` — Python dependency list
