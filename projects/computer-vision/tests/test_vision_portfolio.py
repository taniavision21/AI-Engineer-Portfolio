import base64
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/vision_portfolio.py"
SPEC = importlib.util.spec_from_file_location("vision_portfolio", MODULE_PATH)
vision_portfolio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vision_portfolio)


class VisionPortfolioTests(unittest.TestCase):
    def test_extract_text_uses_output_text_when_present(self):
        response = {"output_text": '{"scene_summary":"ok"}', "output": []}
        self.assertEqual(vision_portfolio.extract_text(response), '{"scene_summary":"ok"}')

    def test_extract_text_falls_back_to_nested_output_message_content(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"scene_summary":"fallback"}'}],
                }
            ]
        }
        self.assertEqual(
            vision_portfolio.extract_text(response),
            '{"scene_summary":"fallback"}',
        )

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

    def test_main_raises_for_non_json_response(self):
        args = Mock(image_url="https://example.com/x.png", image_path=None, model="gpt-4.1-mini")
        with patch.object(vision_portfolio, "parse_args", return_value=args):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                with patch.object(vision_portfolio, "analyze_image", return_value="plain text result"):
                    with self.assertRaises(ValueError):
                        vision_portfolio.main()

    def test_main_prints_formatted_json_for_valid_object_response(self):
        args = Mock(image_url="https://example.com/x.png", image_path=None, model="gpt-4.1-mini")
        valid_response = (
            '{"scene_summary":"ok","visible_products":["water"],'
            '"stock_risks":[],"misplacements":[],"accessibility_notes":[],'
            '"recommended_actions":["restock"]}'
        )
        with patch.object(vision_portfolio, "parse_args", return_value=args):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                with patch.object(vision_portfolio, "analyze_image", return_value=valid_response):
                    stdout = io.StringIO()
                    with patch("sys.stdout", stdout):
                        exit_code = vision_portfolio.main()

        self.assertEqual(exit_code, 0)
        self.assertIn('"scene_summary": "ok"', stdout.getvalue())
        self.assertIn('"recommended_actions": [', stdout.getvalue())

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
        self.assertTrue(kwargs["json"]["text"]["format"]["strict"])

    def test_validate_result_rejects_missing_keys(self):
        with self.assertRaises(ValueError):
            vision_portfolio.validate_result({"scene_summary": "ok"})

    def test_validate_result_rejects_wrong_field_types(self):
        with self.assertRaises(ValueError):
            vision_portfolio.validate_result(
                {
                    "scene_summary": "ok",
                    "visible_products": "water",
                    "stock_risks": [],
                    "misplacements": [],
                    "accessibility_notes": [],
                    "recommended_actions": [],
                }
            )

    def test_main_raises_when_json_schema_validation_fails(self):
        args = Mock(image_url="https://example.com/x.png", image_path=None, model="gpt-4.1-mini")
        invalid_object = (
            '{"scene_summary":"ok","visible_products":"water","stock_risks":[],'
            '"misplacements":[],"accessibility_notes":[],"recommended_actions":[]}'
        )
        with patch.object(vision_portfolio, "parse_args", return_value=args):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
                with patch.object(vision_portfolio, "analyze_image", return_value=invalid_object):
                    with self.assertRaises(ValueError):
                        vision_portfolio.main()


if __name__ == "__main__":
    unittest.main()
