import base64
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = (
    Path("/home/runner/work/AI-Engineer-Portfolio/AI-Engineer-Portfolio")
    / "projects/computer-vision/src/vision_portfolio.py"
)
SPEC = importlib.util.spec_from_file_location("vision_portfolio", MODULE_PATH)
vision_portfolio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vision_portfolio)


class VisionPortfolioTests(unittest.TestCase):
    def test_extract_text_uses_output_text_when_present(self):
        response = {"output_text": '{"scene_summary":"ok"}', "output": []}
        self.assertEqual(vision_portfolio.extract_text(response), '{"scene_summary":"ok"}')

    def test_build_input_from_path_embeds_data_url(self):
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jxXYAAAAASUVORK5CYII="
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(png_bytes)
            temp_path = handle.name

        self.addCleanup(lambda: Path(temp_path).unlink(missing_ok=True))

        payload = vision_portfolio.build_input_from_path(temp_path)
        self.assertTrue(payload[0]["content"][1]["image_url"].startswith("data:image/png;base64,"))

    def test_main_requires_api_key(self):
        with patch.object(vision_portfolio, "parse_args", return_value=Mock(image_url="https://example.com/x.png", image_path=None, model="gpt-4.1-mini")):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(EnvironmentError):
                    vision_portfolio.main()

    def test_main_prints_raw_text_when_response_is_not_json(self):
        args = Mock(image_url="https://example.com/x.png", image_path=None, model="gpt-4.1-mini")
        with patch.object(vision_portfolio, "parse_args", return_value=args):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                with patch.object(vision_portfolio, "analyze_image", return_value="plain text result"):
                    stdout = io.StringIO()
                    with patch("sys.stdout", stdout):
                        exit_code = vision_portfolio.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "plain text result")

    def test_analyze_image_sends_json_schema_format(self):
        mock_response = Mock()
        mock_response.json.return_value = {"output_text": '{"scene_summary":"ok"}'}
        mock_response.raise_for_status.return_value = None

        with patch.object(vision_portfolio.requests, "post", return_value=mock_response) as post_mock:
            result = vision_portfolio.analyze_image(
                input_payload=vision_portfolio.build_input_from_url("https://example.com/x.png"),
                model="gpt-4.1-mini",
                api_key="test-key",
            )

        self.assertEqual(result, '{"scene_summary":"ok"}')
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["json"]["text"]["format"]["type"], "json_schema")


if __name__ == "__main__":
    unittest.main()
