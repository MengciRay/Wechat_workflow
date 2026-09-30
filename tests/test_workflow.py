import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import workflow
import pymupdf
import pymupdf4llm


class WorkflowTests(unittest.TestCase):
    def test_pdf_dependency_import_error_shows_underlying_cause(self):
        original_import = __import__

        def incompatible_numpy(name, *args, **kwargs):
            if name == "pymupdf4llm":
                raise ImportError("NumPy binary incompatible")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=incompatible_numpy):
            with self.assertRaisesRegex(RuntimeError, "NumPy binary incompatible"):
                workflow.extract_pdf(Path("unused.pdf"))

    def test_extract_real_text_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((50, 60), "Cell Research " * 5 + "\n" +
                                 ("Study of protein structure and function.\n" * 24), fontsize=11)
                document.new_page()  # Preserve a blank middle page in citation numbering.
                page = document.new_page()
                page.insert_text((50, 60), "Final page evidence.", fontsize=11)
                document.save(path)
            extracted = workflow.extract_pdf(path)
            self.assertEqual(len(extracted), 3)
            self.assertEqual(extracted[1], "")
            self.assertIn("Final page evidence", extracted[2])
            self.assertIn("Cell Research", extracted[0])

    def test_chunks_keep_every_page_and_page_label(self):
        pages = ["A" * 100, "B" * 100, "C" * 100]
        chunks = workflow.page_chunks(pages, limit=150)
        combined = "".join(chunks)
        for number, letter in enumerate("ABC", 1):
            self.assertIn(f"[[PDF 第 {number} 页]]", combined)
            self.assertEqual(combined.count(letter * 100), 1)

    def test_prepare_writes_preview_and_reuses_unchanged_article(self):
        notes = {"bibliography": {"title": "Paper"}, "findings": [{"point": "发现", "pages": [1]}]}
        article = {"title": "解读", "digest": "卡片简介", "summary": {"text": "核心总结 <X>", "pages": [1]}, "source": {"title": "Paper"},
                   "sections": [{"heading": "主要发现", "paragraphs": [{"text": "发现 <X>", "pages": [1]}]}],
                   "review_flags": []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"fake PDF for mocked extraction")
            with patch.object(workflow, "extract_pdf", return_value=["Study text"]), \
                 patch.object(workflow, "deepseek_json", side_effect=[notes, article, article]) as model:
                result = workflow.prepare(pdf, root / "output", "单细胞组学")
                again = workflow.prepare(pdf, root / "output", "单细胞组学")
            self.assertEqual(result, again)
            self.assertEqual(model.call_count, 3)
            saved = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(saved["article"]["sections"][0]["paragraphs"][0]["pages"], [1])
            self.assertEqual(saved["notes"], [notes])
            self.assertEqual(saved["generation_config"]["extractor"], "pymupdf4llm")
            self.assertIn("Study text", result.with_suffix(".source.md").read_text(encoding="utf-8"))
            self.assertIn("原文第 1 页", result.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertEqual(saved["article"]["summary"], article["summary"])
            self.assertEqual(workflow.article_style_issues(saved["article"]), [])
            # A pipeline/prompt update must regenerate an otherwise identical old cache.
            saved["generation_config"]["pipeline_version"] = 0
            result.write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(workflow, "extract_pdf", return_value=["Study text"]), \
                 patch.object(workflow, "deepseek_json", side_effect=[notes, article, article]) as changed:
                workflow.prepare(pdf, root / "output", "单细胞组学")
            self.assertEqual(changed.call_count, 3)
            preview = result.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn("&lt;X&gt;", preview)
            self.assertIn("原文第 1 页", preview)
            copyable = result.with_suffix(".wechat.html").read_text(encoding="utf-8")
            self.assertIn("&lt;X&gt;", copyable)
            self.assertNotIn("发布前检查", copyable)
            self.assertIn("核心总结 &lt;X&gt;", copyable)
            self.assertLess(copyable.index("核心总结 &lt;X&gt;"), copyable.index("<h2"))
            self.assertNotIn("卡片简介", copyable)
            markdown = result.with_suffix(".md").read_text(encoding="utf-8")
            self.assertLess(markdown.index("核心总结 <X>"), markdown.index("## 主要发现"))
            self.assertNotIn("卡片简介", markdown)

    def test_extract_preserves_markdown_table(self):
        markdown = "# Results\n\n| Group | N |\n| --- | --- |\n| A | 23 |\n\n" + "Evidence " * 100
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "table.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.save(path)
            with patch.object(pymupdf4llm, "to_markdown", return_value=[{"text": markdown}]) as convert:
                self.assertEqual(workflow.extract_pdf(path), [markdown.strip()])
            self.assertTrue(convert.call_args.kwargs["page_chunks"])
            self.assertFalse(convert.call_args.kwargs["use_ocr"])

    def test_blank_and_encrypted_pdf_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            blank, locked = Path(temp) / "blank.pdf", Path(temp) / "locked.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.save(blank)
                document.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                              owner_pw="owner", user_pw="secret")
            with self.assertRaisesRegex(RuntimeError, "OCR"):
                workflow.extract_pdf(blank)
            with self.assertRaisesRegex(RuntimeError, "加密"):
                workflow.extract_pdf(locked)

    def test_long_page_chunking_keeps_all_text_and_bounds(self):
        text = "Row | 23 | Evidence\n" * 80
        chunks = workflow.page_chunks(["", text], limit=150)
        rebuilt = "".join(chunk.split("\n", 2)[2][:-1] for chunk in chunks)
        self.assertEqual(rebuilt, text)
        self.assertTrue(all(len(chunk) <= 150 for chunk in chunks))
        self.assertTrue(all("[[PDF 第 2 页]]" in chunk for chunk in chunks))
        with self.assertRaises(ValueError):
            workflow.page_chunks([text], limit=40)

    def short_article(self, title="解读"):
        return {"title": title, "digest": "卡片简介", "source": {"title": "Paper"},
                "summary": {"text": "这项研究发现了新的候选蛋白，但功能仍需实验验证。", "pages": [1]},
                "sections": [{"kind": "findings", "heading": "主要发现",
                              "paragraphs": [{"text": "有依据的发现。" * 40, "pages": [1]}]}]}

    def test_short_article_is_not_expanded_to_old_minimum(self):
        with patch.object(workflow, "deepseek_json", return_value=self.short_article()) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(workflow.article_style_issues(result), [])
        self.assertLess(workflow.article_stats(result)["body_chars"], 1200)

    def test_full_source_review_is_exported(self):
        draft = self.short_article("初稿")
        reviewed = self.short_article("校核后的标题")
        with patch.object(workflow, "deepseek_json", side_effect=[draft, reviewed]) as model:
            result = workflow.generate_article([], "", 1, source_markdown="[[PDF 第 1 页]] Original table 23")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(result["title"], "校核后的标题")
        self.assertIn("Original table 23", model.call_args.args[1])
        self.assertIn("初稿", model.call_args.args[1])

    def test_overlong_article_accepts_shorter_revision(self):
        draft = self.short_article()
        draft["sections"][0]["paragraphs"][0]["text"] = "长" * 2200
        revised = self.short_article()
        with patch.object(workflow, "deepseek_json", side_effect=[draft, revised]) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 2)
        self.assertLess(workflow.article_stats(result)["body_chars"], 2000)
        self.assertEqual(result["sections"][0]["paragraphs"][0]["text"], revised["sections"][0]["paragraphs"][0]["text"])

    def test_failed_shortening_does_not_export_long_article(self):
        draft = self.short_article()
        draft["sections"][0]["paragraphs"][0]["text"] = "长" * 2200
        with patch.object(workflow, "deepseek_json", return_value=draft) as model:
            with self.assertRaisesRegex(RuntimeError, "精简要求"):
                workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 2)

    def test_summary_is_repaired_without_truncating_sentences(self):
        draft = self.short_article()
        draft["summary"] = {"text": "过长的总结。" * 40, "pages": [True, 0, 2]}
        normalized = workflow.normalize_article(draft, 1)
        self.assertEqual(normalized["summary"]["text"], draft["summary"]["text"])
        self.assertEqual(normalized["summary"]["pages"], [])
        self.assertEqual(len(workflow.article_style_issues(normalized)), 2)
        with patch.object(workflow, "deepseek_json", side_effect=[draft, self.short_article()]) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(workflow.article_style_issues(result), [])

    def test_missing_summary_is_generated(self):
        draft = self.short_article()
        del draft["summary"]
        with patch.object(workflow, "deepseek_json", side_effect=[draft, self.short_article()]):
            result = workflow.generate_article([], "", 1)
        self.assertTrue(result["summary"]["text"])

    def test_methods_are_limited_to_one_short_paragraph(self):
        draft = self.short_article()
        draft["sections"].append({"heading": "研究怎么做", "paragraphs": [
            {"text": "技术参数" * 50, "pages": [1]}, {"text": "第二段方法", "pages": [1]}]})
        normalized = workflow.normalize_article(draft, 1)
        self.assertEqual(normalized["sections"][1]["kind"], "methods")
        self.assertTrue(any("方法描述过多" in issue for issue in workflow.article_style_issues(normalized)))
        with patch.object(workflow, "deepseek_json", side_effect=[draft, self.short_article()]) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(workflow.article_style_issues(result), [])

    def test_revision_cannot_drop_body_citations(self):
        draft = self.short_article()
        draft["sections"][0]["paragraphs"][0]["text"] = "长" * 2200
        revised = self.short_article()
        revised["sections"][0]["paragraphs"][0]["pages"] = []
        with patch.object(workflow, "deepseek_json", side_effect=[draft, revised]):
            with self.assertRaisesRegex(RuntimeError, "丢失.*页码"):
                workflow.generate_article([], "", 1)

    def test_all_wechat_requests_bypass_system_proxy_by_default(self):
        direct = MagicMock()
        direct.open.side_effect = [
            io.BytesIO(b'{"access_token":"test-token"}'),
            io.BytesIO(b'{"total_count":0}'),
            io.BytesIO(b'{"media_id":"cover-id"}'),
        ]
        with patch.dict(workflow.os.environ, {"WECHAT_APP_ID": "test-app", "WECHAT_APP_SECRET": "test-secret"}, clear=True), \
             patch.object(workflow.urllib.request, "build_opener", return_value=direct) as build, \
             patch.object(workflow.urllib.request, "urlopen") as system, \
             patch.object(workflow, "default_cover", return_value=b"test-cover"):
            token = workflow.wechat_token()
            self.assertEqual(workflow.wechat_api("draft/count", token)["total_count"], 0)
            self.assertEqual(workflow.upload_cover(token, None), "cover-id")
        self.assertEqual(direct.open.call_count, 3)
        system.assert_not_called()
        for call in build.call_args_list:
            self.assertEqual(call.args[0].proxies, {})
        paths = [workflow.urllib.parse.urlsplit(call.args[0].full_url).path for call in direct.open.call_args_list]
        self.assertEqual(paths, ["/cgi-bin/token", "/cgi-bin/draft/count", "/cgi-bin/material/add_material"])

    def test_wechat_system_proxy_can_be_explicitly_enabled(self):
        with patch.dict(workflow.os.environ, {"WECHAT_APP_ID": "test-app", "WECHAT_APP_SECRET": "test-secret", "WECHAT_USE_SYSTEM_PROXY": "1"}), \
             patch.object(workflow.urllib.request, "build_opener") as build, \
             patch.object(workflow.urllib.request, "urlopen", return_value=io.BytesIO(b'{"access_token":"test-token"}')) as system:
            self.assertEqual(workflow.wechat_token(), "test-token")
        system.assert_called_once()
        build.assert_not_called()

    def test_deepseek_keeps_system_network_routing(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        with patch.dict(workflow.os.environ, {"DEEPSEEK_API_KEY": "test-key", "WECHAT_USE_SYSTEM_PROXY": "0"}), \
             patch.object(workflow.urllib.request, "build_opener") as build, \
             patch.object(workflow.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as system:
            self.assertEqual(workflow.deepseek_json("system", "user"), {"ok": True})
        system.assert_called_once()
        build.assert_not_called()

    def test_draft_payload_and_receipt_prevent_duplicate_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "article.json"
            path.write_text(json.dumps({"article": {"title": "测试", "digest": "摘要",
                "source": {"doi": "10.1234/example"},
                "sections": [{"heading": "研究问题", "paragraphs": [{"text": "内容", "pages": [2]}]}],
                "review_flags": []}}, ensure_ascii=False), encoding="utf-8")
            with patch.object(workflow, "wechat_token", return_value="token"), \
                 patch.object(workflow, "upload_cover", return_value="cover-id"), \
                 patch.object(workflow, "wechat_api", return_value={"media_id": "draft-id"}) as api:
                first = workflow.create_draft(path)
                second = workflow.create_draft(path)
            self.assertEqual(first, second)
            self.assertEqual(api.call_count, 1)
            payload = api.call_args.args[2]
            self.assertEqual(payload["articles"][0]["thumb_media_id"], "cover-id")
            self.assertEqual(payload["articles"][0]["content_source_url"], "https://doi.org/10.1234/example")


if __name__ == "__main__":
    unittest.main()
