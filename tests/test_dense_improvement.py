"""
정의 전용 Child 추가(Dense v2) 개선 작업 검증 테스트.

- 청킹 단계: 신규 Child 생성과 기존 205개 Child 보존
- 정답 참조 정규화: Golden Set 평가 로직의 조문 경로 비교
- Retriever: 기본 컬렉션 유지와 버전 선택
- Qdrant 상태: v1 205개 / v2 206개, 기존 벡터 동일성
  (Qdrant에 접속할 수 없으면 해당 테스트만 건너뛴다.)
- 실제 Dense 검색 테스트는 Embedding API를 호출하므로
  RUN_RETRIEVAL_TESTS=1 환경변수가 있을 때만 실행한다.
"""

import json
import os
import sys
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.golden_schema import (
    matches,
    parse_reference,
    reference_depth,
    result_reference,
)
from rag.chunker import (
    DEFINITION_ONLY_RULES,
    LAW_EFFECTIVE_DATE,
    LAW_NUMBER,
    LAW_SHORT_NAME,
    XML_PATH,
    extract_definition_term,
    get_main_base_children,
    parse_law,
    validate_chunks,
)
from rag.retriever import COLLECTION_NAME, COLLECTION_NAME_V2, format_article_path

V1_CHUNKS_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks_v1.json"
V2_CHUNKS_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks_v2.json"

ARTICLE2_PARENT_ID = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조"
ARTICLE2_ITEM4_ID = f"{ARTICLE2_PARENT_ID}-p1-i4"
DEFINITION_CHILD_ID = f"{ARTICLE2_ITEM4_ID}-definition"

SUBITEM_LABELS = [
    "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카",
]

def read_xml_item4_content():
    root = ET.parse(XML_PATH).getroot()

    for article in root.findall("./조문/조문단위"):
        if (article.findtext("조문번호") or "").strip() != "2":
            continue

        for paragraph in article.findall("항"):
            for item in paragraph.findall("호"):
                if (item.findtext("호번호") or "").strip() == "4.":
                    return (item.findtext("호내용") or "").strip()

    raise AssertionError("XML에서 제2조제4호를 찾을 수 없습니다.")

class DefinitionOnlyChunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = parse_law()
        cls.parents = cls.result["parents"]
        cls.children = cls.result["children"]
        cls.child_map = {c["id"]: c for c in cls.children}
        cls.xml_item4 = read_xml_item4_content()

    def test_validate_chunks_passes(self):
        self.assertTrue(validate_chunks(self.result))

    def test_counts(self):
        self.assertEqual(len(self.parents), 49)
        self.assertEqual(len(self.children), 206)

    def test_definition_child_exists_exactly_once(self):
        definition_children = [
            c for c in self.children
            if c.get("chunk_role") == "definition_only"
        ]

        self.assertEqual(len(definition_children), 1)
        self.assertEqual(definition_children[0]["id"], DEFINITION_CHILD_ID)

    def test_definition_child_id_is_deterministic(self):
        again = parse_law()
        again_ids = {c["id"] for c in again["children"]}

        self.assertIn(DEFINITION_CHILD_ID, again_ids)
        self.assertEqual(
            {c["id"] for c in self.children},
            again_ids,
        )

    def test_definition_child_source_text_is_xml_original(self):
        child = self.child_map[DEFINITION_CHILD_ID]

        self.assertEqual(child["source_text"], self.xml_item4)

    def test_definition_child_excludes_subitems(self):
        child = self.child_map[DEFINITION_CHILD_ID]

        self.assertNotIn("\n", child["source_text"])

        for label in SUBITEM_LABELS:
            self.assertNotIn(f"{label}. ", child["source_text"])

    def test_definition_child_text_has_search_context(self):
        child = self.child_map[DEFINITION_CHILD_ID]

        self.assertTrue(child["text"].startswith(LAW_SHORT_NAME))
        self.assertIn("제2조(정의)", child["text"])
        self.assertIn("제4호 고영향 인공지능", child["text"])
        self.assertIn(child["source_text"], child["text"])

    def test_definition_child_metadata(self):
        child = self.child_map[DEFINITION_CHILD_ID]

        self.assertEqual(child["parent_id"], ARTICLE2_PARENT_ID)
        self.assertEqual(child["document_type"], "main")
        self.assertEqual(child["article"], "제2조")
        self.assertEqual(child["article_title"], "정의")
        self.assertEqual(child["item"], "4.")
        self.assertEqual(child["subitem"], "")
        self.assertEqual(child["granularity"], "item")
        self.assertEqual(child["chunk_role"], "definition_only")
        self.assertEqual(child["law_number"], LAW_NUMBER)
        self.assertEqual(child["effective_date"], LAW_EFFECTIVE_DATE)

    def test_definition_child_schema_matches_existing(self):
        existing = self.child_map[ARTICLE2_ITEM4_ID]
        definition = self.child_map[DEFINITION_CHILD_ID]

        self.assertTrue(set(existing).issubset(set(definition)))
        self.assertEqual(set(definition) - set(existing), {"chunk_role"})

    def test_item4_full_child_and_subitems_kept(self):
        self.assertIn(ARTICLE2_ITEM4_ID, self.child_map)

        for label in SUBITEM_LABELS:
            self.assertIn(f"{ARTICLE2_ITEM4_ID}-{label}", self.child_map)

    def test_supplementary_children_kept(self):
        supplementary = [
            c for c in self.children
            if c["document_type"] == "supplementary"
        ]

        self.assertEqual(len(supplementary), 8)

    def test_definition_child_excluded_from_base_children(self):
        """
        정의 전용 Child는 원문 재조합 기준이 되는 기본 단위 Child에
        포함되지 않아야 한다. (포함되면 원문이 중복된다.)
        """

        related = [
            c for c in self.children
            if c["parent_id"] == ARTICLE2_PARENT_ID
        ]
        base_children = get_main_base_children(
            "제2조", ARTICLE2_PARENT_ID, related
        )
        base_ids = {c["id"] for c in base_children}

        self.assertNotIn(DEFINITION_CHILD_ID, base_ids)
        self.assertIn(ARTICLE2_ITEM4_ID, base_ids)

    def test_definition_rule_scope(self):
        self.assertEqual(list(DEFINITION_ONLY_RULES), [("제2조", "4")])

    def test_extract_definition_term(self):
        self.assertEqual(
            extract_definition_term(self.xml_item4),
            "고영향 인공지능",
        )
        self.assertEqual(extract_definition_term("따옴표 없음"), "")

class ExistingChunkPreservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not V1_CHUNKS_PATH.exists():
            raise unittest.SkipTest(f"v1 스냅샷이 없습니다: {V1_CHUNKS_PATH}")

        cls.v1_children = json.loads(
            V1_CHUNKS_PATH.read_text(encoding="utf-8")
        )["children"]
        cls.current = parse_law()
        cls.current_map = {c["id"]: c for c in cls.current["children"]}

    def test_v1_snapshot_has_205_children(self):
        self.assertEqual(len(self.v1_children), 205)

    def test_existing_child_ids_all_kept(self):
        for child in self.v1_children:
            self.assertIn(child["id"], self.current_map)

    def test_existing_child_text_unchanged(self):
        for child in self.v1_children:
            current = self.current_map[child["id"]]

            self.assertEqual(current["text"], child["text"])
            self.assertEqual(current["source_text"], child["source_text"])

    def test_only_definition_child_added(self):
        v1_ids = {c["id"] for c in self.v1_children}
        added = set(self.current_map) - v1_ids

        self.assertEqual(added, {DEFINITION_CHILD_ID})

class GoldenReferenceTest(unittest.TestCase):
    def test_parse_article_path(self):
        reference = parse_reference("제2조제4호바목")

        self.assertEqual(reference["article"], "제2조")
        self.assertEqual(reference["item"], "4")
        self.assertEqual(reference["subitem"], "바")
        self.assertEqual(reference_depth(reference), "subitem")

    def test_parse_paragraph_path(self):
        reference = parse_reference("제34조제1항제2호")

        self.assertEqual(reference["article"], "제34조")
        self.assertEqual(reference["paragraph"], "1")
        self.assertEqual(reference["item"], "2")
        self.assertEqual(reference_depth(reference), "item")

    def test_parse_branch_article(self):
        reference = parse_reference("제11조의2")

        self.assertEqual(reference["article"], "제11조의2")
        self.assertEqual(reference_depth(reference), "article")

    def test_parse_chunk_id(self):
        reference = parse_reference(DEFINITION_CHILD_ID)

        self.assertEqual(reference["article"], "제2조")
        self.assertEqual(reference["item"], "4")
        self.assertEqual(reference["subitem"], "")

    def test_item_level_gold_matches_definition_and_subitem_children(self):
        """
        정답이 '제2조제4호'이면 전체 Child, 정의 전용 Child,
        가목~카목 Child가 모두 적중으로 집계되어야 한다.
        """

        gold = parse_reference("제2조제4호")

        for candidate in (
            {"article": "제2조", "paragraph": "", "item": "4.", "subitem": ""},
            {"article": "제2조", "paragraph": "", "item": "4.", "subitem": "바."},
        ):
            self.assertTrue(matches(gold, result_reference(candidate)))

    def test_subitem_level_gold_is_strict(self):
        gold = parse_reference("제2조제4호사목")

        self.assertTrue(matches(gold, result_reference(
            {"article": "제2조", "paragraph": "", "item": "4.", "subitem": "사."}
        )))
        self.assertFalse(matches(gold, result_reference(
            {"article": "제2조", "paragraph": "", "item": "4.", "subitem": "바."}
        )))
        self.assertFalse(matches(gold, result_reference(
            {"article": "제2조", "paragraph": "", "item": "4.", "subitem": ""}
        )))

    def test_circled_paragraph_normalized(self):
        gold = parse_reference("제34조제1항")

        self.assertTrue(matches(gold, result_reference(
            {"article": "제34조", "paragraph": "①", "item": "2.", "subitem": ""}
        )))
        self.assertFalse(matches(gold, result_reference(
            {"article": "제34조", "paragraph": "②", "item": "", "subitem": ""}
        )))

class RetrieverContractTest(unittest.TestCase):
    def test_default_collection_is_v1(self):
        """
        기본 컬렉션이 바뀌면 기존 기능이 예고 없이 v2를 쓰게 되므로
        기본값은 v1이어야 한다.
        """

        self.assertEqual(COLLECTION_NAME, "beopjeong_law_v1")
        self.assertEqual(COLLECTION_NAME_V2, "beopjeong_law_v2")

    def test_dense_search_accepts_collection_name(self):
        import inspect

        from rag.retriever import dense_search

        signature = inspect.signature(dense_search)

        self.assertIn("collection_name", signature.parameters)
        self.assertEqual(
            signature.parameters["collection_name"].default,
            "beopjeong_law_v1",
        )

    def test_format_article_path(self):
        self.assertEqual(
            format_article_path({
                "article": "제34조",
                "paragraph": "①",
                "item": "2.",
                "subitem": "",
            }),
            "제34조제1항제2호",
        )
        self.assertEqual(
            format_article_path({
                "article": "제2조",
                "paragraph": "",
                "item": "4.",
                "subitem": "바.",
            }),
            "제2조제4호바목",
        )
        self.assertEqual(
            format_article_path({
                "article": "제2조",
                "paragraph": "",
                "item": "4.",
                "subitem": "",
                "chunk_role": "definition_only",
            }),
            "제2조제4호(정의)",
        )

def get_client_or_skip():
    try:
        from common.qdrant import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()

        return client
    except Exception as error:
        raise unittest.SkipTest(f"Qdrant 접속 불가: {error}")

class QdrantCollectionStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = get_client_or_skip()

    def test_v1_has_205_points(self):
        self.assertTrue(self.client.collection_exists("beopjeong_law_v1"))
        self.assertEqual(
            self.client.get_collection("beopjeong_law_v1").points_count,
            205,
        )

    def test_v2_has_206_points(self):
        if not self.client.collection_exists("beopjeong_law_v2"):
            self.skipTest("v2 컬렉션이 아직 구축되지 않았습니다.")

        self.assertEqual(
            self.client.get_collection("beopjeong_law_v2").points_count,
            206,
        )

    def test_vector_config_matches(self):
        if not self.client.collection_exists("beopjeong_law_v2"):
            self.skipTest("v2 컬렉션이 아직 구축되지 않았습니다.")

        v1 = self.client.get_collection(
            "beopjeong_law_v1"
        ).config.params.vectors
        v2 = self.client.get_collection(
            "beopjeong_law_v2"
        ).config.params.vectors

        self.assertEqual(v1.size, v2.size)
        self.assertEqual(v1.size, 1536)
        self.assertEqual(v1.distance, v2.distance)

    def test_existing_vectors_identical(self):
        if not self.client.collection_exists("beopjeong_law_v2"):
            self.skipTest("v2 컬렉션이 아직 구축되지 않았습니다.")

        from scripts.build_dense_v2 import read_all_points

        v1_points = read_all_points(self.client, "beopjeong_law_v1")
        v2_points = {
            str(point.id): point
            for point in read_all_points(self.client, "beopjeong_law_v2")
        }

        self.assertEqual(len(v1_points), 205)
        self.assertEqual(len(v2_points), 206)

        for point in v1_points:
            copied = v2_points.get(str(point.id))

            self.assertIsNotNone(copied, msg=f"미복사 Point: {point.id}")
            self.assertEqual(list(copied.vector), list(point.vector))
            self.assertEqual(copied.payload, point.payload)

    def test_definition_child_point_present_in_v2_only(self):
        if not self.client.collection_exists("beopjeong_law_v2"):
            self.skipTest("v2 컬렉션이 아직 구축되지 않았습니다.")

        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, DEFINITION_CHILD_ID)
        )

        in_v2 = self.client.retrieve(
            collection_name="beopjeong_law_v2",
            ids=[point_id],
            with_payload=True,
        )
        in_v1 = self.client.retrieve(
            collection_name="beopjeong_law_v1",
            ids=[point_id],
            with_payload=True,
        )

        self.assertEqual(len(in_v2), 1)
        self.assertEqual(len(in_v1), 0)
        self.assertEqual(
            in_v2[0].payload["chunk_role"], "definition_only"
        )

@unittest.skipUnless(
    os.getenv("RUN_RETRIEVAL_TESTS") == "1",
    "Embedding API를 호출하므로 RUN_RETRIEVAL_TESTS=1일 때만 실행한다.",
)
class DenseRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = get_client_or_skip()

        if not cls.client.collection_exists("beopjeong_law_v2"):
            raise unittest.SkipTest("v2 컬렉션이 없습니다.")

        from common.ai_model import get_embedding_model
        from rag.retriever import dense_search

        cls.embeddings = get_embedding_model()
        cls.dense_search = staticmethod(dense_search)

    def search(self, collection_name, top_k=30):
        from rag.retriever import dense_search

        return dense_search(
            "고영향 인공지능이란 무엇인가요?",
            top_k=top_k,
            collection_name=collection_name,
            embeddings=self.embeddings,
            client=self.client,
        )

    def test_definition_question_ranks_definition_child_first_in_v2(self):
        results = self.search("beopjeong_law_v2", top_k=5)

        self.assertEqual(results[0]["id"], DEFINITION_CHILD_ID)
        self.assertEqual(results[0]["rank"], 1)

    def test_definition_child_absent_in_v1(self):
        results = self.search("beopjeong_law_v1", top_k=30)
        ids = [r["id"] for r in results]

        self.assertNotIn(DEFINITION_CHILD_ID, ids)

    def test_v1_results_order_preserved_below_new_child(self):
        """
        v2는 기존 벡터를 그대로 복사했으므로, 신규 Child를 제외하면
        v1과 동일한 순서가 유지되어야 한다.
        """

        v1_ids = [r["id"] for r in self.search("beopjeong_law_v1", top_k=20)]
        v2_ids = [
            r["id"] for r in self.search("beopjeong_law_v2", top_k=21)
            if r["id"] != DEFINITION_CHILD_ID
        ]

        self.assertEqual(v1_ids, v2_ids[:len(v1_ids)])

if __name__ == "__main__":
    unittest.main()
