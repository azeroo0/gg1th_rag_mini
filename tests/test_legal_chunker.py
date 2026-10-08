import re
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.chunker import (
    LAW_EFFECTIVE_DATE,
    LAW_NUMBER,
    get_main_base_children,
    normalize_whitespace,
    parse_law,
    validate_chunks,
)

class LegalChunkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = parse_law()
        cls.parents = cls.result["parents"]
        cls.children = cls.result["children"]
        cls.parent_map = {p["id"]: p for p in cls.parents}
        cls.child_ids = {c["id"] for c in cls.children}

    def test_validate_chunks_passes(self):
        self.assertTrue(validate_chunks(self.result))

    def test_main_parent_count(self):
        main_parents = [
            p for p in self.parents if p["document_type"] == "main"
        ]
        self.assertEqual(len(main_parents), 46)

    def test_supplementary_parent_count(self):
        supplementary_parents = [
            p for p in self.parents if p["document_type"] == "supplementary"
        ]
        self.assertEqual(len(supplementary_parents), 3)

    def test_no_duplicate_ids(self):
        parent_ids = [p["id"] for p in self.parents]
        child_ids = [c["id"] for c in self.children]
        self.assertEqual(len(parent_ids), len(set(parent_ids)))
        self.assertEqual(len(child_ids), len(set(child_ids)))

    def test_all_parent_id_references_valid(self):
        for child in self.children:
            self.assertIn(child["parent_id"], self.parent_map)

    def test_no_empty_text(self):
        for child in self.children:
            self.assertTrue(child["text"].strip())

    def test_article2_item4_full_child_kept(self):
        full_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조-p1-i4"
        self.assertIn(full_id, self.child_ids)

    def test_article2_item4_subitems_added(self):
        item4_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조-p1-i4"
        labels = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카"]
        expected_ids = [f"{item4_id}-{label}" for label in labels]

        for expected_id in expected_ids:
            self.assertIn(expected_id, self.child_ids)

        self.assertEqual(len(expected_ids), 11)

    def test_article34_paragraph1_full_child_kept(self):
        paragraph1_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조-p1"
        self.assertIn(paragraph1_id, self.child_ids)

    def test_article34_paragraph1_items_added(self):
        parent_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조"
        expected_ids = [f"{parent_id}-p1-i{i}" for i in range(1, 7)]

        for expected_id in expected_ids:
            self.assertIn(expected_id, self.child_ids)

    def test_subitem_source_text_matches_xml_original(self):
        """
        목 단위 세부 Child의 source_text는 수정 없이 원문 그대로여야 한다.
        """

        item4_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조-p1-i4"
        child_map = {c["id"]: c for c in self.children}

        ga_child = child_map[f"{item4_id}-가"]
        self.assertIn("에너지법", ga_child["source_text"])
        self.assertTrue(ga_child["source_text"].startswith("가."))

    def test_item_child_context_includes_paragraph_condition(self):
        """
        제34조제1항 세부 Child의 text에는 상위 항의 조건이 포함되어야 한다.
        """

        item1_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조-p1-i1"
        child_map = {c["id"]: c for c in self.children}

        item1 = child_map[item1_id]
        self.assertIn("대통령령으로 정하는 바에 따라 이행하여야 한다", item1["text"])
        self.assertEqual(item1["source_text"], "1. 위험관리방안의 수립ㆍ운영")

    def test_main_body_reconstructs_to_original_text(self):
        for parent in self.parents:
            if parent["document_type"] != "main":
                continue

            related_children = [
                c for c in self.children if c["parent_id"] == parent["id"]
            ]
            base_children = get_main_base_children(
                parent["article"], parent["id"], related_children
            )

            reconstructed = "\n".join(c["source_text"] for c in base_children)

            header = (
                f"{parent['article']}({parent['article_title']})"
                if parent["article_title"]
                else parent["article"]
            )
            body = parent["text"]

            if body.startswith(header + "\n"):
                body = body[len(header) + 1:]

            self.assertEqual(
                normalize_whitespace(reconstructed),
                normalize_whitespace(body),
                msg=f"원문 불일치: {parent['id']}",
            )

    def test_supplementary_reconstructs_to_original_text(self):
        for parent in self.parents:
            if parent["document_type"] != "supplementary":
                continue

            related_children = [
                c for c in self.children if c["parent_id"] == parent["id"]
            ]

            reconstructed = "\n".join(c["source_text"] for c in related_children)

            self.assertEqual(
                normalize_whitespace(reconstructed),
                normalize_whitespace(parent["text"]),
                msg=f"부칙 원문 불일치: {parent['id']}",
            )

    def test_metadata_consistency(self):
        law_names = {c["law_name"] for c in self.parents + self.children}
        law_numbers = {c["law_number"] for c in self.parents + self.children}
        main_effective_dates = {
            c["effective_date"]
            for c in self.parents + self.children
            if c["document_type"] == "main"
        }

        self.assertEqual(len(law_names), 1)
        self.assertEqual(len(law_numbers), 1)
        self.assertEqual(len(main_effective_dates), 1)

    def test_all_children_have_required_schema_fields(self):
        required_fields = [
            "id",
            "parent_id",
            "chunk_type",
            "document_type",
            "law_name",
            "law_number",
            "effective_date",
            "chapter",
            "section",
            "article",
            "article_title",
            "paragraph",
            "item",
            "subitem",
            "granularity",
            "text",
            "source_text",
        ]

        for child in self.children:
            for field in required_fields:
                self.assertIn(field, child, msg=f"{child.get('id')} missing {field}")

if __name__ == "__main__":
    unittest.main()
