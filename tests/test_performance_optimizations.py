import concurrent.futures
from PIL import Image

from app.agents.verifier import VerificationRouteDecision, determine_verification_route
from app.documents.ingestion import PageImageCache
from app.documents.ocr import DocumentOcrService, DocumentRepresentation, OcrLine, OcrPage
from app.services.gemini_client import GeminiDocumentClient
from tests.test_accounting_workflow import settings


def dummy_image(color="white"):
    return Image.new("RGB", (10, 10), color=color)


def test_page_image_cache_isolation_and_scoping():
    cache = PageImageCache(capacity=5)
    img_a = dummy_image("red")
    img_b = dummy_image("blue")

    cache.put("doc_hash_A", 150, [img_a])
    cache.put("doc_hash_B", 150, [img_b])

    retrieved_a = cache.get("doc_hash_A", 150)
    retrieved_b = cache.get("doc_hash_B", 150)

    assert retrieved_a is not None
    assert retrieved_b is not None
    assert retrieved_a[0].getpixel((0, 0)) == (255, 0, 0)
    assert retrieved_b[0].getpixel((0, 0)) == (0, 0, 255)

    # Cross-document isolation: Doc A key cannot retrieve Doc B
    assert cache.get("doc_hash_A", 200) is None
    assert cache.get("doc_hash_C", 150) is None


def test_page_image_cache_bounded_capacity_lru():
    cache = PageImageCache(capacity=3)
    for i in range(5):
        cache.put(f"hash_{i}", 150, [dummy_image()])

    # Only the 3 most recent entries should remain
    assert cache.size() <= 3
    assert cache.get("hash_0", 150) is None
    assert cache.get("hash_1", 150) is None
    assert cache.get("hash_4", 150) is not None


def test_page_image_cache_thread_safety():
    cache = PageImageCache(capacity=10)

    def worker(idx):
        img = dummy_image()
        cache.put(f"thread_doc_{idx}", 150, [img])
        got = cache.get(f"thread_doc_{idx}", 150)
        return got is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(20)))

    assert all(results)
    assert cache.size() <= 10


def test_ocr_service_reuse_hits_and_misses():
    service = DocumentOcrService(poppler_path=None, primary=None, secondary=None)
    assert service.hits == 0
    assert service.misses == 0

    # Cache miss
    assert service.cached("sha256_unknown") is None
    assert service.misses == 1
    assert service.hits == 0

    # Store a representation
    rep = DocumentRepresentation(
        manifest={"sha256": "sha256_known"},
        pages=[],
        engine="RapidOCR",
        dpi=150,
    )
    service.remember(rep)

    # Cache hit
    cached_rep = service.cached("sha256_known")
    assert cached_rep is not None
    assert service.hits == 1
    assert service.misses == 1

    status = service.status()
    assert status["cache_hits"] == 1
    assert status["cache_misses"] == 1


def test_verification_routing_digital_vs_handwritten():
    # 1. Clean digital representation with high confidence
    clean_lines = [
        OcrLine(text=f"Line {i} details ₹1000", bbox=(10, 10 * i, 100, 10 * i + 8), confidence=0.98, page=1, engine="RapidOCR")
        for i in range(25)
    ]
    clean_page = OcrPage(page_number=1, width=200, height=300, lines=clean_lines, engine="RapidOCR", script="PRINTED")
    clean_rep = DocumentRepresentation(
        manifest={"pages": 1},
        pages=[clean_page],
        engine="RapidOCR",
        dpi=150,
    )

    decision = determine_verification_route(clean_rep, {"rows": []}, has_images=True)
    assert decision.mode == "OCR_STRUCTURED"
    assert not decision.is_handwritten_or_uncertain
    assert "Clean digital document" in decision.reason

    # 2. Handwritten / noisy representation with low confidence
    noisy_lines = [
        OcrLine(text="unclear scribbled note", bbox=(10, 10, 50, 20), confidence=0.60, page=1, engine="RapidOCR")
    ]
    noisy_page = OcrPage(page_number=1, width=200, height=300, lines=noisy_lines, engine="RapidOCR", script="PRINTED")
    noisy_rep = DocumentRepresentation(
        manifest={"pages": 1},
        pages=[noisy_page],
        engine="RapidOCR",
        dpi=150,
    )
    decision_noisy = determine_verification_route(noisy_rep, {"rows": []}, has_images=True)
    assert decision_noisy.mode == "MULTIMODAL"
    assert decision_noisy.is_handwritten_or_uncertain

    # 3. Representation with warnings
    warn_page = OcrPage(page_number=1, width=200, height=300, lines=clean_lines, engine="RapidOCR", script="PRINTED")
    warn_rep = DocumentRepresentation(
        manifest={"pages": 1},
        pages=[warn_page],
        engine="RapidOCR",
        dpi=150,
        warnings=["low_contrast"],
    )
    decision_warn = determine_verification_route(warn_rep, {"rows": []}, has_images=True)
    assert decision_warn.mode == "MULTIMODAL"
    assert "warnings" in decision_warn.reason


def test_gemini_client_instrumentation():
    client = GeminiDocumentClient(settings())
    assert hasattr(client, "call_history")
    # Bounded: process-wide diagnostics must not grow without limit.
    assert client.call_history.maxlen is not None
