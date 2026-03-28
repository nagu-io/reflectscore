import unittest

from retrieval import chunk_text, chunk_tokens, split_code_sections, tokenize


class RetrievalTests(unittest.TestCase):
    def test_chunk_tokens_uses_overlap(self):
        tokens = [f"tok{i}" for i in range(260)]
        chunks = chunk_tokens(tokens, chunk_size=200, overlap=50)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][-50:], chunks[1][:50])

    def test_split_code_sections_extracts_file_headers(self):
        text = "# file: a.py\ndef first():\n    pass\n# file: b.py\ndef second():\n    pass\n"
        sections = split_code_sections(text)
        self.assertEqual(sections[0][0], "a.py")
        self.assertEqual(sections[1][0], "b.py")

    def test_tokenize_splits_on_whitespace(self):
        self.assertEqual(tokenize("a  b\n c"), ["a", "b", "c"])

    def test_chunk_text_uses_tokenizer_when_available(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return list(range(6))

            def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                del skip_special_tokens, clean_up_tokenization_spaces
                return " ".join(f"tok{token}" for token in tokens)

        chunks = chunk_text("ignored", chunk_size=4, overlap=2, tokenizer=FakeTokenizer())
        self.assertEqual(chunks, ["tok0 tok1 tok2 tok3", "tok2 tok3 tok4 tok5"])


if __name__ == "__main__":
    unittest.main()
