"""Versioned local accounting knowledge base with TF-IDF retrieval (no external service)."""

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "accounting"


class KnowledgeBaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    title: str
    text: str
    tags: tuple[str, ...]
    source: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text, "tags": list(self.tags), "source": self.source}


class KnowledgeBase:
    def __init__(self, path: Path = DEFAULT_KB_PATH):
        self.path = Path(path)
        manifest_file = self.path / "manifest.json"
        if not manifest_file.is_file():
            raise KnowledgeBaseError(f"Knowledge base manifest not found at {manifest_file}")
        self.manifest = self._load("manifest.json")
        self.data = {name: self._load(name) for name in self.manifest["files"]}
        self.entries = self._entries()
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._matrix = self._vectorizer.fit_transform([f"{e.title} {e.text} {' '.join(e.tags)}" for e in self.entries])

    def _load(self, name: str) -> dict:
        try:
            return json.loads((self.path / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"Cannot load knowledge base file {name}: {exc}") from exc

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def document_types(self) -> dict:
        return self.data["document_types.json"]

    @property
    def vocabulary(self) -> dict:
        return self.data["vocabulary.json"]

    @property
    def gst(self) -> dict:
        return self.data["gst.json"]

    @property
    def tds(self) -> dict:
        return self.data["tds.json"]

    @property
    def expense_categories(self) -> dict[str, list[str]]:
        return self.data["expense_categories.json"]["categories"]

    @property
    def rule_catalog(self) -> dict:
        return self.data["validation_rules.json"]

    @property
    def parameters(self) -> dict:
        return self.rule_catalog["parameters"]

    def rule(self, rule_id: str) -> dict:
        for rule in self.rule_catalog["rules"]:
            if rule["id"] == rule_id:
                return rule
        raise KnowledgeBaseError(f"Unknown rule {rule_id}")

    def policy(self, policy_id: str) -> dict:
        for policy in self.data["policies.json"]["policies"]:
            if policy["id"] == policy_id:
                return policy
        raise KnowledgeBaseError(f"Unknown policy {policy_id}")

    def policies(self) -> list[dict]:
        return list(self.data["policies.json"]["policies"])

    def gst_rates_on(self, on: date | None) -> list[float]:
        on = on or date.today()
        for slab in self.gst["rate_slabs"]:
            start = date.fromisoformat(slab["effective_from"])
            end = date.fromisoformat(slab["effective_to"]) if slab["effective_to"] else date.max
            if start <= on <= end:
                return [float(rate) for rate in slab["rates"]]
        return []

    def tds_rates(self) -> list[float]:
        rates = {float(rate) for section in self.tds["sections"] for rate in section["rates"]}
        rates.add(float(self.tds["no_pan_rate"]))
        return sorted(rates)

    def suggest_category(self, text: str | None) -> str | None:
        lowered = (text or "").lower()
        best, best_len = None, 0
        for category, keywords in self.expense_categories.items():
            for keyword in keywords:
                if keyword in lowered and len(keyword) > best_len:
                    best, best_len = category, len(keyword)
        return best

    def _entries(self) -> list[KnowledgeEntry]:
        entries = [
            KnowledgeEntry(e["id"], e["title"], e["text"], tuple(e.get("tags", [])), "concepts.json")
            for e in self.data["concepts.json"]["entries"]
        ]
        for source in ("gst.json", "tds.json"):
            entries += [
                KnowledgeEntry(r["id"], r["id"].replace("_", " ").title(), r["text"], (source.split(".")[0],), source)
                for r in self.data[source]["rules"]
            ]
        entries += [
            KnowledgeEntry(r["id"], r["id"].replace("_", " ").title(), r["description"], tuple(r["applies_to"]), "validation_rules.json")
            for r in self.rule_catalog["rules"]
        ]
        entries += [
            KnowledgeEntry(p["id"], p["id"].replace("_", " ").title(), f"{p['status']}: {p['question']}", ("policy",), "policies.json")
            for p in self.policies()
        ]
        return entries

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeEntry]:
        if not query.strip():
            return []
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [self.entries[i] for i in ranked if scores[i] > 0]

    def summary(self) -> dict:
        return {
            "version": self.version,
            "updated": self.manifest["updated"],
            "entries": len(self.entries),
            "document_types": list(self.document_types["types"].keys()),
            "rules": len(self.rule_catalog["rules"]),
            "policies_required": [p["id"] for p in self.policies() if p["status"] == "POLICY_REQUIRED"],
        }


@lru_cache(maxsize=1)
def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
