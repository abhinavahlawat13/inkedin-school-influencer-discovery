from pydantic import BaseModel, Field
from typing import List, Optional

class ProfileModel(BaseModel):
    id: str
    name: str
    linkedin_url: str
    current_role: str
    organization: str
    location: str
    experience_years: int = Field(default=0)
    persona_type: str  # e.g., "School Director", "Education Consultant", "Former Principal", "K-12 Entrepreneur"
    ecosystem_indicators: List[str]
    bio_summary: str
    key_affiliations: List[str]

    def to_embedding_text(self) -> str:
        """Flattens the profile into a rich contextual document for semantic similarity matching."""
        indicators = " ".join(self.ecosystem_indicators)
        affiliations = " ".join(self.key_affiliations)
        return (
            f"Role: {self.current_role}. Organization: {self.organization}. "
            f"Persona: {self.persona_type}. Summary: {self.bio_summary}. "
            f"Ecosystem Indicators: {indicators}. Affiliations: {affiliations}."
        )

class MatchResult(BaseModel):
    profile: ProfileModel
    similarity_score: float
    relevance_score: float
    match_reasons: List[str]