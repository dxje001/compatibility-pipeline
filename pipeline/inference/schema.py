"""
Input schema for UI questionnaire responses.

Defines the data structures for questionnaire answers that match
the approved UI composition from ui_questionnaire_proposal.md.

UI Composition (10 items total):
- Personality (P1-P5): 5 OCEAN Likert questions
- Interests (I1-I5): Religion, About Me, Lifestyle, Family, Education
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ReligionChoice(Enum):
    """Religion categorical options."""
    CHRISTIANITY_SERIOUS = "christianity_serious"
    CHRISTIANITY_CASUAL = "christianity_casual"
    CATHOLICISM_SERIOUS = "catholicism_serious"
    CATHOLICISM_CASUAL = "catholicism_casual"
    JUDAISM_SERIOUS = "judaism_serious"
    JUDAISM_CASUAL = "judaism_casual"
    ISLAM_SERIOUS = "islam_serious"
    ISLAM_CASUAL = "islam_casual"
    HINDUISM_SERIOUS = "hinduism_serious"
    HINDUISM_CASUAL = "hinduism_casual"
    BUDDHISM_SERIOUS = "buddhism_serious"
    BUDDHISM_CASUAL = "buddhism_casual"
    ATHEISM = "atheism"
    AGNOSTICISM = "agnosticism"
    SPIRITUAL = "other_spiritual"
    OTHER = "other"
    UNKNOWN = "unknown"


class LifestyleChoice(Enum):
    """Lifestyle habits categorical options (merged: drinks/smokes/drugs)."""
    VERY_HEALTHY = "very_healthy"           # No drinking, smoking, or drugs
    MOSTLY_HEALTHY = "mostly_healthy"       # Social drinking only
    MODERATE = "moderate"                   # Social drinking, occasional smoking
    RELAXED = "relaxed"                     # Regular drinking/smoking
    UNKNOWN = "unknown"


class FamilyChoice(Enum):
    """Family & relationship categorical options (merged: status/offspring)."""
    SINGLE_NO_KIDS_WANTS = "single_no_kids_wants"
    SINGLE_NO_KIDS_DOESNT_WANT = "single_no_kids_doesnt_want"
    SINGLE_HAS_KIDS = "single_has_kids"
    SINGLE_UNDECIDED = "single_undecided"
    RELATIONSHIP_NO_KIDS = "relationship_no_kids"
    RELATIONSHIP_HAS_KIDS = "relationship_has_kids"
    UNKNOWN = "unknown"


class EducationChoice(Enum):
    """Education level categorical options."""
    HIGH_SCHOOL = "high_school"
    SOME_COLLEGE = "some_college"
    TWO_YEAR_COLLEGE = "two_year_college"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD_LAW_MD = "phd_law_md"
    TRADE_SCHOOL = "trade_school"
    UNKNOWN = "unknown"


@dataclass
class PersonalityAnswers:
    """
    Personality questionnaire answers (P1-P5).

    All answers are on a 1-5 Likert scale:
    1 = Strongly Disagree
    2 = Disagree
    3 = Neutral
    4 = Agree
    5 = Strongly Agree

    Attributes:
        extraversion: P1 - "I feel comfortable around people"
        agreeableness: P2 - "I am interested in other people's problems"
        conscientiousness: P3 - "I pay attention to details"
        openness: P4 - "I have a vivid imagination"
        neuroticism: P5 - "I get stressed out easily"
    """
    extraversion: int  # P1: 1-5
    agreeableness: int  # P2: 1-5
    conscientiousness: int  # P3: 1-5
    openness: int  # P4: 1-5
    neuroticism: int  # P5: 1-5

    def __post_init__(self):
        """Validate Likert scale bounds."""
        for attr in ["extraversion", "agreeableness", "conscientiousness",
                     "openness", "neuroticism"]:
            val = getattr(self, attr)
            if not isinstance(val, int) or not 1 <= val <= 5:
                raise ValueError(f"{attr} must be an integer between 1 and 5, got {val}")

    def to_ocean_vector(self) -> List[float]:
        """
        Convert to OCEAN vector in standard order.

        Returns:
            List of 5 floats: [extraversion, neuroticism, agreeableness,
                              conscientiousness, openness]
        """
        return [
            float(self.extraversion),
            float(self.neuroticism),
            float(self.agreeableness),
            float(self.conscientiousness),
            float(self.openness)
        ]

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "conscientiousness": self.conscientiousness,
            "openness": self.openness,
            "neuroticism": self.neuroticism
        }


@dataclass
class InterestsAnswers:
    """
    Interests questionnaire answers (I1-I5).

    Attributes:
        religion: I1 - Religious background and seriousness
        about_me: I2 - Free-text describing interests, lifestyle, values
        lifestyle: I3 - Lifestyle habits (drinking, smoking, drugs)
        family: I4 - Family and relationship status
        education: I5 - Highest education level
    """
    religion: ReligionChoice  # I1: Categorical
    about_me: str  # I2: Free-text
    lifestyle: LifestyleChoice  # I3: Categorical (merged)
    family: FamilyChoice  # I4: Categorical (merged)
    education: EducationChoice  # I5: Categorical

    def __post_init__(self):
        """Validate and convert string inputs to enums if needed."""
        if isinstance(self.religion, str):
            self.religion = ReligionChoice(self.religion)
        if isinstance(self.lifestyle, str):
            self.lifestyle = LifestyleChoice(self.lifestyle)
        if isinstance(self.family, str):
            self.family = FamilyChoice(self.family)
        if isinstance(self.education, str):
            self.education = EducationChoice(self.education)

        if not isinstance(self.about_me, str):
            raise ValueError(f"about_me must be a string, got {type(self.about_me)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string values."""
        return {
            "religion": self.religion.value,
            "about_me": self.about_me,
            "lifestyle": self.lifestyle.value,
            "family": self.family.value,
            "education": self.education.value
        }


@dataclass
class QuestionnaireResponse:
    """
    Complete questionnaire response for one person.

    Combines personality (P1-P5) and interests (I1-I5) answers.

    Attributes:
        personality: PersonalityAnswers instance
        interests: InterestsAnswers instance
        person_id: Optional identifier for the person
    """
    personality: PersonalityAnswers
    interests: InterestsAnswers
    person_id: Optional[str] = None

    def __post_init__(self):
        """Validate nested objects."""
        if isinstance(self.personality, dict):
            self.personality = PersonalityAnswers(**self.personality)
        if isinstance(self.interests, dict):
            self.interests = InterestsAnswers(**self.interests)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "person_id": self.person_id,
            "personality": self.personality.to_dict(),
            "interests": self.interests.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionnaireResponse":
        """Create from dictionary."""
        return cls(
            personality=PersonalityAnswers(**data["personality"]),
            interests=InterestsAnswers(**data["interests"]),
            person_id=data.get("person_id")
        )


@dataclass
class CompatibilityResult:
    """
    Result of compatibility scoring.

    Attributes:
        personality_score: Score from personality model [0, 1]
        interests_score: Score from interests model [0, 1]
        final_score: Fused final compatibility score [0, 1]
        breakdown: Optional detailed breakdown of score components
    """
    personality_score: float
    interests_score: float
    final_score: float
    breakdown: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "personality_score": self.personality_score,
            "interests_score": self.interests_score,
            "final_score": self.final_score
        }
        if self.breakdown:
            result["breakdown"] = self.breakdown
        return result
