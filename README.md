# AI Engineer Portfolio

This repository now includes a production-style computer vision portfolio project built around a current multimodal vision API.

## Featured Project

### Vision-Based Retail Shelf Auditor

A computer vision workflow that reviews a store shelf image and returns:

- product visibility observations
- empty shelf / low-stock signals
- misplaced item notes
- accessibility-friendly scene summaries
- structured JSON output for downstream automation

The project uses the latest OpenAI Responses API pattern for image understanding and demonstrates how an AI engineer can turn raw visual input into structured business insights.

## Project Location

- [`projects/computer-vision/README.md`](./projects/computer-vision/README.md)

## Skills Demonstrated

- multimodal prompt engineering
- image-to-JSON extraction
- API integration and environment-based secret handling
- portfolio-ready project documentation
- production-minded CLI workflow design

## Quick Start

```bash
git clone https://github.com/taniavision21/AI-Engineer-Portfolio.git
cd AI-Engineer-Portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r projects/computer-vision/requirements.txt
python projects/computer-vision/src/vision_portfolio.py --help
```