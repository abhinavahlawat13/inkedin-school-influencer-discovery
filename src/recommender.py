import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import ProfileModel, MatchResult


class InfluencerRecommender:
    def __init__(self, candidates_path: str = "data/candidates.json"):
        self.candidates_path = Path(candidates_path)
        self.candidates: List[ProfileModel] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.candidate_embeddings = None
        self._load_candidates()

    def _load_candidates(self):
        """Loads and parses candidate profiles from JSON."""
        if not self.candidates_path.exists():
            raise FileNotFoundError(f"Candidate file not found at {self.candidates_path}")
        
        with open(self.candidates_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.candidates = [ProfileModel(**item) for item in data]
        
        # Fit vectorizer on the candidate corpus
        corpus = [c.to_embedding_text() for c in self.candidates]
        self.candidate_embeddings = self.vectorizer.fit_transform(corpus)

    def _compute_indicator_overlap(self, ref: ProfileModel, candidate: ProfileModel) -> Tuple[float, List[str]]:
        """Calculates overlap between ecosystem indicators and key affiliations."""
        ref_tokens = set([w.lower() for item in (ref.ecosystem_indicators + ref.key_affiliations) for w in item.split()])
        cand_tokens = set([w.lower() for item in (candidate.ecosystem_indicators + candidate.key_affiliations) for w in item.split()])
        
        # Common educational keywords to ignore as distinct matches
        stop_words = {"school", "schools", "education", "educational", "india", "and", "the", "of", "in", "for"}
        meaningful_overlap = (ref_tokens & cand_tokens) - stop_words

        # Specific institutional/governance overlap checks
        shared_entities = []
        for ref_aff in (ref.key_affiliations + ref.ecosystem_indicators):
            for cand_aff in (candidate.key_affiliations + candidate.ecosystem_indicators):
                ref_clean = ref_aff.lower()
                cand_clean = cand_aff.lower()
                if (ref_clean in cand_clean or cand_clean in ref_clean) and len(ref_clean) > 3:
                    if ref_aff not in shared_entities:
                        shared_entities.append(ref_aff)

        overlap_score = min(1.0, len(meaningful_overlap) / 4.0)
        return overlap_score, shared_entities

    def _generate_match_reasons(
        self,
        ref: ProfileModel,
        candidate: ProfileModel,
        sim_score: float,
        shared_entities: List[str]
    ) -> List[str]:
        """Generates dynamic, human-readable match explanations."""
        reasons = []

        # 1. Persona alignment
        if ref.persona_type == candidate.persona_type:
            reasons.append(f"Direct persona alignment as a fellow **{candidate.persona_type}**.")
        else:
            reasons.append(f"Complementary school ecosystem role: {ref.persona_type} ↔ {candidate.persona_type}.")

        # 2. Shared institutional networks or councils
        if shared_entities:
            unique_shared = list(dict.fromkeys(shared_entities))[:2]
            reasons.append(f"Shared institutional networks / governance ties: {', '.join(unique_shared)}.")

        # 3. Experience and leadership scope
        if abs(ref.experience_years - candidate.experience_years) <= 7:
            reasons.append(f"Comparable leadership seniority (~{candidate.experience_years} years in K-12 schooling).")
        
        # 4. Semantic contextual relevance
        if sim_score >= 0.35:
            reasons.append("High contextual similarity across pedagogical focus, policy advisory, and school governance.")
        else:
            reasons.append("Relevance driven by shared ecosystem indicators and leadership networks.")

        return reasons[:3]

    def rank_similar_profiles(self, reference_profile: ProfileModel, top_k: int = 25) -> List[MatchResult]:
        """
        Executes the end-to-end matching workflow:
        1. Embeds the reference profile.
        2. Computes cosine similarity against candidate corpus.
        3. Calculates persona & indicator bonuses.
        4. Ranks candidates and attaches explainability breakdown.
        """
        ref_text = reference_profile.to_embedding_text()
        ref_vector = self.vectorizer.transform([ref_text])

        # Raw Cosine Similarity (Range: 0 to 1)
        cosine_sims = cosine_similarity(ref_vector, self.candidate_embeddings)[0]

        results: List[MatchResult] = []

        for idx, candidate in enumerate(self.candidates):
            # Skip if comparing to the exact same profile
            if candidate.id == reference_profile.id or candidate.name.lower() == reference_profile.name.lower():
                continue

            raw_sim = float(cosine_sims[idx])

            # Persona alignment weight
            persona_bonus = 1.0 if candidate.persona_type == reference_profile.persona_type else 0.4

            # Ecosystem & affiliation overlap
            overlap_score, shared_entities = self._compute_indicator_overlap(reference_profile, candidate)

            # Composite Relevance Scoring Formula
            # 70% Semantic Vector Similarity + 15% Persona Alignment + 15% Network/Affiliation Overlap
            relevance = (0.70 * raw_sim) + (0.15 * persona_bonus) + (0.15 * overlap_score)
            
            # Normalize to 0-100 percentage scale for UI display
            relevance_percentage = round(min(99.0, max(40.0, relevance * 100)), 1)
            raw_sim_percentage = round(raw_sim * 100, 1)

            reasons = self._generate_match_reasons(reference_profile, candidate, raw_sim, shared_entities)

            results.append(
                MatchResult(
                    profile=candidate,
                    similarity_score=raw_sim_percentage,
                    relevance_score=relevance_percentage,
                    match_reasons=reasons
                )
            )

        # Sort descending by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]