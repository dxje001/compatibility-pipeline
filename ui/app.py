"""
Compatibility Scoring UI

A Streamlit application for computing compatibility scores
between two persons based on their questionnaire responses.

Design: Premium psychology/relationship clinic aesthetic
- Soft neutral palette (off-white, charcoal, subtle teal accent)
- Large hero score display
- Clean whitespace and typography

Run with: streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from pipeline.inference import CompatibilityPredictor, QuestionnaireResponse
from pipeline.inference.schema import (
    PersonalityAnswers,
    InterestsAnswers,
    ReligionChoice,
    LifestyleChoice,
    FamilyChoice,
    EducationChoice,
)

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_TEXT_LENGTH = 200
ARTIFACTS_DIR = project_root / "artifacts" / "runs" / "seed_11"

# =============================================================================
# DESIGN SYSTEM - Colors & Styles
# =============================================================================

COLORS = {
    "background": "#FAFAFA",
    "card_bg": "#FFFFFF",
    "text_primary": "#2D3748",
    "text_secondary": "#718096",
    "text_muted": "#A0AEC0",
    "accent": "#319795",  # Subtle teal
    "accent_light": "#E6FFFA",
    "border": "#E2E8F0",
    "success": "#48BB78",
    "warning": "#ED8936",
    "error": "#F56565",
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for clinic-grade aesthetic."""
    st.markdown(f"""
    <style>
        /* Base styles */
        .stApp {{
            background-color: {COLORS['background']};
        }}

        /* Typography */
        h1, h2, h3 {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600 !important;
        }}

        p, .stMarkdown {{
            color: {COLORS['text_secondary']};
        }}

        /* Card container */
        .card {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}

        .card-header {{
            color: {COLORS['text_primary']};
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid {COLORS['border']};
        }}

        /* Hero score display */
        .hero-score-container {{
            text-align: center;
            padding: 2.5rem 1rem;
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 16px;
            margin: 1.5rem 0;
        }}

        .hero-score {{
            font-size: 4.5rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            line-height: 1;
            margin-bottom: 0.5rem;
        }}

        .hero-score-label {{
            font-size: 1rem;
            color: {COLORS['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
        }}

        .hero-score-bar {{
            height: 6px;
            background: {COLORS['border']};
            border-radius: 3px;
            margin: 1.5rem auto 0;
            max-width: 280px;
            overflow: hidden;
        }}

        .hero-score-fill {{
            height: 100%;
            background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_light']});
            border-radius: 3px;
            transition: width 0.5s ease;
        }}

        /* Subscore display */
        .subscore-container {{
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid {COLORS['border']};
        }}

        .subscore {{
            text-align: center;
        }}

        .subscore-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
        }}

        .subscore-label {{
            font-size: 0.8rem;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Confidence badge */
        .confidence-badge {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .confidence-high {{
            background: #C6F6D5;
            color: #276749;
        }}

        .confidence-medium {{
            background: #FEEBC8;
            color: #975A16;
        }}

        .confidence-low {{
            background: #FED7D7;
            color: #C53030;
        }}

        /* Section divider */
        .section-divider {{
            height: 1px;
            background: {COLORS['border']};
            margin: 2rem 0;
        }}

        /* Question styling */
        .question-text {{
            color: {COLORS['text_primary']};
            font-weight: 500;
            margin-bottom: 0.5rem;
        }}

        .question-help {{
            color: {COLORS['text_muted']};
            font-size: 0.8rem;
        }}

        /* Character counter */
        .char-counter {{
            font-size: 0.8rem;
            color: {COLORS['text_muted']};
            text-align: right;
            margin-top: 0.25rem;
        }}

        .char-counter.warning {{
            color: {COLORS['warning']};
        }}

        .char-counter.success {{
            color: {COLORS['success']};
        }}

        /* Privacy note */
        .privacy-note {{
            font-size: 0.75rem;
            color: {COLORS['text_muted']};
            font-style: italic;
            margin-top: 0.5rem;
        }}

        /* Info banner */
        .info-banner {{
            background: {COLORS['accent_light']};
            border-left: 3px solid {COLORS['accent']};
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }}

        .info-banner p {{
            color: {COLORS['text_primary']};
            margin: 0;
            font-size: 0.9rem;
        }}

        /* Warning banner (soft) */
        .warning-banner {{
            background: #FFFAF0;
            border-left: 3px solid {COLORS['warning']};
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }}

        .warning-banner p {{
            color: #975A16;
            margin: 0;
            font-size: 0.85rem;
        }}

        /* Likert scale display */
        .likert-label {{
            font-size: 0.8rem;
            color: {COLORS['accent']};
            font-weight: 500;
        }}

        /* Button styling */
        .stButton > button {{
            background: {COLORS['accent']} !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            background: #2C7A7B !important;
            box-shadow: 0 4px 12px rgba(49, 151, 149, 0.3) !important;
        }}

        /* Expander styling */
        .streamlit-expanderHeader {{
            font-size: 0.9rem !important;
            color: {COLORS['text_secondary']} !important;
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Reduce default padding */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# ENUM DISPLAY MAPPINGS
# =============================================================================

RELIGION_OPTIONS = {
    "Christianity (serious)": ReligionChoice.CHRISTIANITY_SERIOUS,
    "Christianity (casual)": ReligionChoice.CHRISTIANITY_CASUAL,
    "Catholicism (serious)": ReligionChoice.CATHOLICISM_SERIOUS,
    "Catholicism (casual)": ReligionChoice.CATHOLICISM_CASUAL,
    "Judaism (serious)": ReligionChoice.JUDAISM_SERIOUS,
    "Judaism (casual)": ReligionChoice.JUDAISM_CASUAL,
    "Islam (serious)": ReligionChoice.ISLAM_SERIOUS,
    "Islam (casual)": ReligionChoice.ISLAM_CASUAL,
    "Hinduism (serious)": ReligionChoice.HINDUISM_SERIOUS,
    "Hinduism (casual)": ReligionChoice.HINDUISM_CASUAL,
    "Buddhism (serious)": ReligionChoice.BUDDHISM_SERIOUS,
    "Buddhism (casual)": ReligionChoice.BUDDHISM_CASUAL,
    "Atheism": ReligionChoice.ATHEISM,
    "Agnosticism": ReligionChoice.AGNOSTICISM,
    "Spiritual (other)": ReligionChoice.SPIRITUAL,
    "Other": ReligionChoice.OTHER,
    "Prefer not to say": ReligionChoice.UNKNOWN,
}

LIFESTYLE_OPTIONS = {
    "Very healthy (no drinking, smoking, or drugs)": LifestyleChoice.VERY_HEALTHY,
    "Mostly healthy (social drinking only)": LifestyleChoice.MOSTLY_HEALTHY,
    "Moderate (social drinking, occasional smoking)": LifestyleChoice.MODERATE,
    "Relaxed (regular drinking/smoking)": LifestyleChoice.RELAXED,
    "Prefer not to say": LifestyleChoice.UNKNOWN,
}

FAMILY_OPTIONS = {
    "Single, no kids, wants kids": FamilyChoice.SINGLE_NO_KIDS_WANTS,
    "Single, no kids, doesn't want kids": FamilyChoice.SINGLE_NO_KIDS_DOESNT_WANT,
    "Single, has kids": FamilyChoice.SINGLE_HAS_KIDS,
    "Single, undecided about kids": FamilyChoice.SINGLE_UNDECIDED,
    "In a relationship, no kids": FamilyChoice.RELATIONSHIP_NO_KIDS,
    "In a relationship, has kids": FamilyChoice.RELATIONSHIP_HAS_KIDS,
    "Prefer not to say": FamilyChoice.UNKNOWN,
}

EDUCATION_OPTIONS = {
    "High school": EducationChoice.HIGH_SCHOOL,
    "Some college": EducationChoice.SOME_COLLEGE,
    "Two-year college": EducationChoice.TWO_YEAR_COLLEGE,
    "Bachelor's degree": EducationChoice.BACHELORS,
    "Master's degree": EducationChoice.MASTERS,
    "PhD / Law / MD": EducationChoice.PHD_LAW_MD,
    "Trade school": EducationChoice.TRADE_SCHOOL,
    "Prefer not to say": EducationChoice.UNKNOWN,
}

LIKERT_LABELS = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly Agree",
}


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

@st.cache_resource
def load_predictor():
    """Load and cache the predictor."""
    if not ARTIFACTS_DIR.exists():
        st.error(f"Model artifacts not found. Please ensure the training pipeline has been run.")
        st.stop()
    return CompatibilityPredictor(str(ARTIFACTS_DIR))


def render_header():
    """Render the page header with title and description."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0.5rem;">Compatibility Assessment</h1>
        <p style="font-size: 1.1rem; color: #718096; max-width: 600px; margin: 0 auto;">
            A research-based compatibility estimate using personality traits and shared interests
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="info-banner">
        <p>
            <strong>Note:</strong> This tool provides an estimate based on psychological research models.
            Results are for informational purposes only and should not be considered definitive compatibility predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_personality_section(person_label: str, key_prefix: str) -> dict:
    """Render personality questions with improved styling."""
    st.markdown(f"""
    <div class="card-header">Personality Profile</div>
    """, unsafe_allow_html=True)

    st.caption("Rate how much you agree with each statement")

    questions = {
        "extraversion": ("I feel comfortable around people", "Measures sociability and energy from social interaction"),
        "agreeableness": ("I am interested in other people's problems", "Measures empathy and cooperation"),
        "conscientiousness": ("I pay attention to details", "Measures organization and dependability"),
        "openness": ("I have a vivid imagination", "Measures creativity and openness to new experiences"),
        "neuroticism": ("I get stressed out easily", "Measures emotional sensitivity"),
    }

    answers = {}

    for key, (question, tooltip) in questions.items():
        st.markdown(f'<p class="question-text">{question}</p>', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            answers[key] = st.slider(
                label=question,
                min_value=1,
                max_value=5,
                value=3,
                key=f"{key_prefix}_{key}",
                format="%d",
                help=tooltip,
                label_visibility="collapsed",
            )
        with col2:
            st.markdown(f'<span class="likert-label">{LIKERT_LABELS[answers[key]]}</span>',
                       unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    return answers


def render_interests_section(person_label: str, key_prefix: str) -> dict:
    """Render interests questions with improved styling."""
    st.markdown(f"""
    <div class="card-header">Interests & Background</div>
    """, unsafe_allow_html=True)

    answers = {}

    # Religion
    st.markdown('<p class="question-text">Religious or spiritual background</p>', unsafe_allow_html=True)
    religion_label = st.selectbox(
        "Religious background",
        options=list(RELIGION_OPTIONS.keys()),
        key=f"{key_prefix}_religion",
        label_visibility="collapsed",
    )
    answers["religion"] = RELIGION_OPTIONS[religion_label]

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # About Me (free text)
    st.markdown('<p class="question-text">About you</p>', unsafe_allow_html=True)
    st.markdown('<p class="question-help">Describe your interests, lifestyle, values, and what matters to you</p>',
                unsafe_allow_html=True)

    about_me = st.text_area(
        "About you",
        height=120,
        key=f"{key_prefix}_about_me",
        placeholder="Share your hobbies, what you enjoy doing, your values, and what you're looking for in a connection...",
        label_visibility="collapsed",
    )
    answers["about_me"] = about_me

    # Character counter with styling
    text_len = len(about_me)
    if text_len == 0:
        counter_class = ""
        counter_text = f"0 / {MIN_TEXT_LENGTH} characters (minimum)"
    elif text_len < MIN_TEXT_LENGTH:
        counter_class = "warning"
        remaining = MIN_TEXT_LENGTH - text_len
        counter_text = f"{text_len} / {MIN_TEXT_LENGTH} characters ({remaining} more for best results)"
    else:
        counter_class = "success"
        counter_text = f"{text_len} characters"

    st.markdown(f'<p class="char-counter {counter_class}">{counter_text}</p>', unsafe_allow_html=True)
    st.markdown('<p class="privacy-note">Your responses are not stored or shared.</p>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Lifestyle
    st.markdown('<p class="question-text">Lifestyle habits</p>', unsafe_allow_html=True)
    lifestyle_label = st.selectbox(
        "Lifestyle habits",
        options=list(LIFESTYLE_OPTIONS.keys()),
        key=f"{key_prefix}_lifestyle",
        label_visibility="collapsed",
    )
    answers["lifestyle"] = LIFESTYLE_OPTIONS[lifestyle_label]

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Family
    st.markdown('<p class="question-text">Family situation</p>', unsafe_allow_html=True)
    family_label = st.selectbox(
        "Family situation",
        options=list(FAMILY_OPTIONS.keys()),
        key=f"{key_prefix}_family",
        label_visibility="collapsed",
    )
    answers["family"] = FAMILY_OPTIONS[family_label]

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # Education
    st.markdown('<p class="question-text">Education level</p>', unsafe_allow_html=True)
    education_label = st.selectbox(
        "Education level",
        options=list(EDUCATION_OPTIONS.keys()),
        key=f"{key_prefix}_education",
        label_visibility="collapsed",
    )
    answers["education"] = EDUCATION_OPTIONS[education_label]

    return answers


def validate_responses(personality_a, interests_a, personality_b, interests_b) -> tuple:
    """Validate questionnaire responses."""
    errors = []
    warnings = []

    text_a_len = len(interests_a["about_me"])
    text_b_len = len(interests_b["about_me"])

    if text_a_len == 0:
        errors.append("Person A: Please fill in the 'About you' section")
    elif text_a_len < MIN_TEXT_LENGTH:
        warnings.append(f"Person A: Adding more detail ({MIN_TEXT_LENGTH - text_a_len} more characters) may improve accuracy")

    if text_b_len == 0:
        errors.append("Person B: Please fill in the 'About you' section")
    elif text_b_len < MIN_TEXT_LENGTH:
        warnings.append(f"Person B: Adding more detail ({MIN_TEXT_LENGTH - text_b_len} more characters) may improve accuracy")

    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def compute_confidence_level(interests_a, interests_b) -> tuple:
    """Compute confidence level based on input completeness."""
    text_a_len = len(interests_a["about_me"])
    text_b_len = len(interests_b["about_me"])

    unknown_count = 0
    for answers in [interests_a, interests_b]:
        if answers["religion"] == ReligionChoice.UNKNOWN:
            unknown_count += 1
        if answers["lifestyle"] == LifestyleChoice.UNKNOWN:
            unknown_count += 1
        if answers["family"] == FamilyChoice.UNKNOWN:
            unknown_count += 1
        if answers["education"] == EducationChoice.UNKNOWN:
            unknown_count += 1

    if text_a_len >= MIN_TEXT_LENGTH and text_b_len >= MIN_TEXT_LENGTH and unknown_count == 0:
        return "High", "Complete responses provided"
    elif text_a_len >= MIN_TEXT_LENGTH // 2 and text_b_len >= MIN_TEXT_LENGTH // 2 and unknown_count <= 2:
        return "Medium", "Some additional detail could improve accuracy"
    else:
        return "Low", "Limited information may affect accuracy"


def render_results(result, confidence_level, confidence_note):
    """Render the results section with hero score display."""

    # Hero score
    score_percent = int(result.final_score * 100)
    bar_width = max(5, score_percent)  # Minimum visible width

    st.markdown(f"""
    <div class="hero-score-container">
        <div class="hero-score-label">Compatibility Estimate</div>
        <div class="hero-score">{score_percent}%</div>
        <div class="hero-score-bar">
            <div class="hero-score-fill" style="width: {bar_width}%"></div>
        </div>
        <div class="subscore-container">
            <div class="subscore">
                <div class="subscore-value">{int(result.personality_score * 100)}%</div>
                <div class="subscore-label">Personality</div>
            </div>
            <div class="subscore">
                <div class="subscore-value">{int(result.interests_score * 100)}%</div>
                <div class="subscore-label">Interests</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence badge
    badge_class = f"confidence-{confidence_level.lower()}"
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="confidence-badge {badge_class}">Confidence: {confidence_level}</span>
        <p style="font-size: 0.85rem; color: #718096; margin-top: 0.5rem;">{confidence_note}</p>
    </div>
    """, unsafe_allow_html=True)

    # Interpretation
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    if result.final_score >= 0.7:
        interpretation = "Your profiles indicate strong alignment in personality traits and/or shared interests."
        interpretation_style = f"background: #C6F6D5; border-left: 3px solid {COLORS['success']};"
    elif result.final_score >= 0.5:
        interpretation = "Your profiles show moderate compatibility with some shared characteristics and some differences."
        interpretation_style = f"background: {COLORS['accent_light']}; border-left: 3px solid {COLORS['accent']};"
    elif result.final_score >= 0.3:
        interpretation = "Your profiles suggest some notable differences in personality or interests."
        interpretation_style = f"background: #FEEBC8; border-left: 3px solid {COLORS['warning']};"
    else:
        interpretation = "Your profiles indicate significant differences in personality traits and interests."
        interpretation_style = f"background: #FED7D7; border-left: 3px solid {COLORS['error']};"

    st.markdown(f"""
    <div style="{interpretation_style} padding: 1rem 1.25rem; border-radius: 0 8px 8px 0;">
        <p style="color: {COLORS['text_primary']}; margin: 0; font-size: 0.95rem;">{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)

    # Technical details expander
    if result.breakdown:
        with st.expander("View technical details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Model Parameters**
                - Fusion weight (α): {result.breakdown['fusion_alpha']}
                - Personality contribution: {result.breakdown['personality_contribution']:.4f}
                - Interests contribution: {result.breakdown['interests_contribution']:.4f}
                """)
            with col2:
                st.markdown(f"""
                **Analysis**
                - Primary factor: {result.breakdown['dominant_model'].title()}
                - Personality score: {result.personality_score:.4f}
                - Interests score: {result.interests_score:.4f}
                """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Compatibility Assessment",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject custom CSS
    inject_custom_css()

    # Load predictor early
    predictor = load_predictor()

    # Header
    render_header()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Two-column layout for Person A and Person B
    col_a, col_spacer, col_b = st.columns([1, 0.05, 1])

    with col_a:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #2D3748;">Person A</h3>
        """, unsafe_allow_html=True)
        personality_a = render_personality_section("Person A", "a")
        st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)
        interests_a = render_interests_section("Person A", "a")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_spacer:
        st.markdown("")  # Visual separator

    with col_b:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #2D3748;">Person B</h3>
        """, unsafe_allow_html=True)
        personality_b = render_personality_section("Person B", "b")
        st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)
        interests_b = render_interests_section("Person B", "b")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Compute button - centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        compute_clicked = st.button(
            "Compute Compatibility",
            type="primary",
            use_container_width=True,
        )

    # Results section
    if compute_clicked:
        # Validate
        is_valid, warnings, errors = validate_responses(
            personality_a, interests_a, personality_b, interests_b
        )

        # Show errors elegantly
        for error in errors:
            st.markdown(f"""
            <div style="background: #FED7D7; border-left: 3px solid {COLORS['error']};
                        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;">
                <p style="color: #C53030; margin: 0; font-size: 0.9rem;">{error}</p>
            </div>
            """, unsafe_allow_html=True)

        # Show warnings elegantly
        for warning in warnings:
            st.markdown(f"""
            <div class="warning-banner">
                <p>{warning}</p>
            </div>
            """, unsafe_allow_html=True)

        if not is_valid:
            st.stop()

        # Compute scores
        try:
            person_a = QuestionnaireResponse(
                personality=PersonalityAnswers(**personality_a),
                interests=InterestsAnswers(**interests_a),
                person_id="person_a"
            )

            person_b = QuestionnaireResponse(
                personality=PersonalityAnswers(**personality_b),
                interests=InterestsAnswers(**interests_b),
                person_id="person_b"
            )

            with st.spinner(""):
                result = predictor.predict(person_a, person_b, return_breakdown=True)

            confidence_level, confidence_note = compute_confidence_level(interests_a, interests_b)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            render_results(result, confidence_level, confidence_note)

        except Exception as e:
            st.markdown(f"""
            <div style="background: #FED7D7; border-left: 3px solid {COLORS['error']};
                        padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                <p style="color: #C53030; margin: 0;"><strong>Error:</strong> Unable to compute compatibility. Please try again.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox("Show technical details"):
                st.exception(e)


if __name__ == "__main__":
    main()
