import datetime as dt
import unittest

from langchain_core.documents import Document

from chat_tools import (
    convert_units,
    date_math,
    eval_expression,
    execute_tool,
    hash_text,
    list_pdfs,
    normalize_tool_calls,
    parse_date,
    read_pdf_page,
    search_pdfs,
)
from harness import try_direct_math, system_prompt


class FakeSession:
    def __init__(self, chunks):
        self.chunks = chunks
        self.hybrid_retriever = None
        self.bm25_retriever = None


class ChatToolsTests(unittest.TestCase):
    def test_percent_of(self):
        ok, text = eval_expression("15% of 240")
        self.assertTrue(ok)
        self.assertTrue(text.endswith("= 36"))

    def test_word_percent_via_tool_alias(self):
        text, docs = execute_tool("calculator", {"query": "what is 15% of 847"})
        self.assertEqual(docs, [])
        self.assertTrue(text.endswith("= 127.05") or text.endswith("= 127.050"))

    def test_sqrt_and_compare(self):
        ok, text = eval_expression("sqrt(16)+2")
        self.assertTrue(ok)
        self.assertTrue(text.endswith("= 6"))
        ok, text = eval_expression("0.15*240 > 30")
        self.assertTrue(ok)
        self.assertTrue(text.endswith("= True"))

    def test_rejects_code(self):
        ok, text = eval_expression("__import__('os').system('echo hi')")
        self.assertFalse(ok)
        self.assertIn("Could not evaluate", text)

    def test_direct_math_still_works(self):
        self.assertEqual(try_direct_math("12*8"), "12*8 = 96")
        self.assertIsNone(try_direct_math("15% of 240"))

    def test_temp_and_length(self):
        self.assertIn("= 212", convert_units({"value": 100, "from_unit": "C", "to_unit": "F"}))
        miles = convert_units({"value": 1, "from": "mile", "to": "km"})
        self.assertIn("1.609344", miles)

    def test_hours_to_minutes(self):
        self.assertIn("= 120", convert_units({"value": 2, "from_unit": "hours", "to_unit": "min"}))

    def test_date_math_diff_and_shift(self):
        diff = date_math({"start": "2026-01-01", "end": "2026-01-11"})
        self.assertIn("10 days", diff)
        shifted = date_math({"start": "2026-08-24", "days": 90})
        self.assertIn("November 22, 2026", shifted)
        self.assertEqual(parse_date("today"), dt.datetime.now().astimezone().date())

    def test_hash_and_clock_aliases(self):
        digest, _ = execute_tool("hash_text", {"text": "cloak", "algorithm": "sha256"})
        self.assertTrue(digest.startswith("sha256: "))
        self.assertEqual(digest.split()[1], hash_text("cloak").split()[1])

    def test_pdf_tools(self):
        session = FakeSession(
            [
                Document(
                    page_content="Either party may terminate this NDA with 30 days notice.",
                    metadata={"source": "nda.pdf", "page": 4, "content_hash": "abc"},
                ),
                Document(
                    page_content="Confidential information excludes public facts.",
                    metadata={"source": "nda.pdf", "page": 2, "content_hash": "abc"},
                ),
            ]
        )
        listed = list_pdfs(session)
        self.assertIn("nda.pdf", listed)
        self.assertIn("pages 2–4", listed)
        text, docs = search_pdfs(session, "terminate notice")
        self.assertTrue(docs)
        self.assertIn("30 days", text)
        page, hits = read_pdf_page(session, "nda", 4)
        self.assertTrue(hits)
        self.assertIn("terminate", page.lower())
        empty, _ = search_pdfs(FakeSession([]), "terminate")
        self.assertIn("No PDFs", empty)

    def test_normalize_openai_tool_calls(self):
        class Msg:
            tool_calls = []
            additional_kwargs = {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        },
                    }
                ]
            }

        calls = normalize_tool_calls(Msg())
        self.assertEqual(calls[0]["name"], "calculator")
        self.assertEqual(calls[0]["args"]["expression"], "2+2")

    def test_system_prompt_lists_pdfs(self):
        text = system_prompt(["nda.pdf"])
        self.assertIn("Attached PDFs: nda.pdf", text)
        self.assertIn("calculator", text)
        self.assertIn("Do not write Python to calculate", text)


if __name__ == "__main__":
    unittest.main()
