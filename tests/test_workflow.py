import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import workflow
import pymupdf
import pymupdf4llm


class WorkflowTests(unittest.TestCase):
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
        article = {"title": "解读", "digest": "摘要", "source": {"title": "Paper"},
                   "sections": [{"heading": "主要发现", "paragraphs": [{"text": "发现 <X>", "pages": [1]}]}],
                   "review_flags": []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"fake PDF for mocked extraction")
            with patch.object(workflow, "extract_pdf", return_value=["Study text"]), \
                 patch.object(workflow, "deepseek_json", side_effect=[notes, article, article, article]) as model:
                result = workflow.prepare(pdf, root / "output", "单细胞组学")
                again = workflow.prepare(pdf, root / "output", "单细胞组学")
            self.assertEqual(result, again)
            self.assertEqual(model.call_count, 4)
            saved = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(saved["article"]["sections"][0]["paragraphs"][0]["pages"], [1])
            self.assertEqual(saved["notes"], [notes])
            self.assertEqual(saved["generation_config"]["extractor"], "pymupdf4llm")
            self.assertIn("Study text", result.with_suffix(".source.md").read_text(encoding="utf-8"))
            self.assertIn("原文第 1 页", result.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertTrue(any("精读完整性检查" in flag for flag in saved["article"]["review_flags"]))
            # A pipeline/prompt update must regenerate an otherwise identical old cache.
            saved["generation_config"]["pipeline_version"] = 0
            result.write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(workflow, "extract_pdf", return_value=["Study text"]), \
                 patch.object(workflow, "deepseek_json", side_effect=[notes, article, article, article]) as changed:
                workflow.prepare(pdf, root / "output", "单细胞组学")
            self.assertEqual(changed.call_count, 4)
            preview = result.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn("&lt;X&gt;", preview)
            self.assertIn("原文第 1 页", preview)
            copyable = result.with_suffix(".wechat.html").read_text(encoding="utf-8")
            self.assertIn("&lt;X&gt;", copyable)
            self.assertNotIn("发布前检查", copyable)

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

    def test_detailed_article_avoids_unnecessary_revision(self):
        article = {"title": "精读", "source": {"title": "Paper"}, "sections": [
            {"heading": str(i), "paragraphs": [{"text": "证据" * 100, "pages": [1]} for _ in range(3)]}
            for i in range(6)]}
        with patch.object(workflow, "deepseek_json", return_value=article) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(workflow.detail_gaps(result), [])

    def test_full_source_review_is_exported(self):
        draft = {"title": "初稿", "source": {"title": "Paper"}, "sections": [
            {"heading": str(i), "paragraphs": [{"text": "证据" * 100, "pages": [1]} for _ in range(3)]}
            for i in range(6)]}
        reviewed = {**draft, "title": "校核后的标题"}
        with patch.object(workflow, "deepseek_json", side_effect=[draft, reviewed]) as model:
            result = workflow.generate_article([], "", 1, source_markdown="[[PDF 第 1 页]] Original table 23")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(result["title"], "校核后的标题")
        self.assertIn("Original table 23", model.call_args.args[1])
        self.assertIn("初稿", model.call_args.args[1])

    def test_revision_with_missing_citations_keeps_first_draft(self):
        first = {"title": "精读", "source": {"title": "Paper"}, "sections": [
            {"heading": "结果", "paragraphs": [{"text": "有依据的发现", "pages": [1]}]}]}
        revised = {"title": "精读", "source": {"title": "Paper"}, "sections": [
            {"heading": str(i), "paragraphs": [{"text": "证据" * 100, "pages": []} for _ in range(3)]}
            for i in range(6)]}
        with patch.object(workflow, "deepseek_json", side_effect=[first, revised]) as model:
            result = workflow.generate_article([], "", 1)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(result["sections"][0]["paragraphs"][0]["text"], "有依据的发现")

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
