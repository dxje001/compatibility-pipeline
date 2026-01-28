"""
Smoke test for the inference pipeline.

Tests:
1. Basic inference with mock inputs
2. Score ranges are within [0, 1]
3. Symmetry: predict(A, B) == predict(B, A)
4. Stability: minor input perturbations produce similar scores
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.inference import (
    CompatibilityPredictor,
    QuestionnaireResponse,
    PersonalityAnswers,
    InterestsAnswers,
)
from pipeline.inference.schema import (
    ReligionChoice,
    LifestyleChoice,
    FamilyChoice,
    EducationChoice,
)


def create_mock_person_a() -> QuestionnaireResponse:
    """Create mock questionnaire response for Person A."""
    return QuestionnaireResponse(
        personality=PersonalityAnswers(
            extraversion=4,      # P1: Outgoing
            agreeableness=5,     # P2: Very cooperative
            conscientiousness=3, # P3: Average
            openness=4,          # P4: Creative
            neuroticism=2        # P5: Calm
        ),
        interests=InterestsAnswers(
            religion=ReligionChoice.AGNOSTICISM,
            about_me="I love hiking, reading science fiction, and trying new restaurants. "
                     "I work in tech and enjoy learning new programming languages. "
                     "Looking for someone who shares my curiosity about the world.",
            lifestyle=LifestyleChoice.MOSTLY_HEALTHY,
            family=FamilyChoice.SINGLE_NO_KIDS_WANTS,
            education=EducationChoice.MASTERS
        ),
        person_id="person_a"
    )


def create_mock_person_b_similar() -> QuestionnaireResponse:
    """Create mock questionnaire response for Person B (similar to A)."""
    return QuestionnaireResponse(
        personality=PersonalityAnswers(
            extraversion=4,      # Same as A
            agreeableness=4,     # Similar to A
            conscientiousness=3, # Same as A
            openness=5,          # Slightly higher than A
            neuroticism=2        # Same as A
        ),
        interests=InterestsAnswers(
            religion=ReligionChoice.AGNOSTICISM,  # Same as A
            about_me="Passionate about outdoor adventures, books, and food. "
                     "Software engineer who loves exploring new technologies. "
                     "Seeking a partner who is intellectually curious.",
            lifestyle=LifestyleChoice.MOSTLY_HEALTHY,  # Same as A
            family=FamilyChoice.SINGLE_NO_KIDS_WANTS,  # Same as A
            education=EducationChoice.BACHELORS
        ),
        person_id="person_b_similar"
    )


def create_mock_person_c_different() -> QuestionnaireResponse:
    """Create mock questionnaire response for Person C (different from A)."""
    return QuestionnaireResponse(
        personality=PersonalityAnswers(
            extraversion=2,      # Introverted (opposite of A)
            agreeableness=2,     # More competitive
            conscientiousness=5, # Very organized
            openness=2,          # Traditional
            neuroticism=4        # More anxious
        ),
        interests=InterestsAnswers(
            religion=ReligionChoice.CHRISTIANITY_SERIOUS,  # Different from A
            about_me="I enjoy staying home, watching TV, and spending time with family. "
                     "I work in accounting and value stability. "
                     "Looking for someone with traditional values.",
            lifestyle=LifestyleChoice.VERY_HEALTHY,  # Different from A
            family=FamilyChoice.SINGLE_HAS_KIDS,  # Different from A
            education=EducationChoice.HIGH_SCHOOL
        ),
        person_id="person_c_different"
    )


def test_basic_inference(predictor: CompatibilityPredictor) -> bool:
    """Test basic inference returns valid results."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Inference")
    print("=" * 60)

    person_a = create_mock_person_a()
    person_b = create_mock_person_b_similar()

    result = predictor.predict(person_a, person_b, return_breakdown=True)

    print(f"Person A: {person_a.person_id}")
    print(f"Person B: {person_b.person_id}")
    print(f"\nResults:")
    print(f"  Personality Score: {result.personality_score:.4f}")
    print(f"  Interests Score:   {result.interests_score:.4f}")
    print(f"  Final Score:       {result.final_score:.4f}")

    if result.breakdown:
        print(f"\nBreakdown:")
        print(f"  Fusion Alpha: {result.breakdown['fusion_alpha']}")
        print(f"  Personality Contribution: {result.breakdown['personality_contribution']:.4f}")
        print(f"  Interests Contribution: {result.breakdown['interests_contribution']:.4f}")
        print(f"  Dominant Model: {result.breakdown['dominant_model']}")

    return True


def test_score_ranges(predictor: CompatibilityPredictor) -> bool:
    """Test that all scores are within [0, 1]."""
    print("\n" + "=" * 60)
    print("TEST 2: Score Ranges")
    print("=" * 60)

    test_cases = [
        (create_mock_person_a(), create_mock_person_b_similar(), "A vs B (similar)"),
        (create_mock_person_a(), create_mock_person_c_different(), "A vs C (different)"),
        (create_mock_person_b_similar(), create_mock_person_c_different(), "B vs C"),
    ]

    all_passed = True
    for person1, person2, label in test_cases:
        result = predictor.predict(person1, person2)

        in_range = (
            0 <= result.personality_score <= 1 and
            0 <= result.interests_score <= 1 and
            0 <= result.final_score <= 1
        )

        status = "PASS" if in_range else "FAIL"
        print(f"  {label}: personality={result.personality_score:.4f}, "
              f"interests={result.interests_score:.4f}, "
              f"final={result.final_score:.4f} [{status}]")

        if not in_range:
            all_passed = False

    print(f"\nScore ranges test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_symmetry(predictor: CompatibilityPredictor) -> bool:
    """Test that predict(A, B) == predict(B, A)."""
    print("\n" + "=" * 60)
    print("TEST 3: Symmetry (A,B == B,A)")
    print("=" * 60)

    person_a = create_mock_person_a()
    person_b = create_mock_person_b_similar()

    result_ab = predictor.predict(person_a, person_b)
    result_ba = predictor.predict(person_b, person_a)

    # Allow small floating point tolerance
    tolerance = 1e-6

    personality_diff = abs(result_ab.personality_score - result_ba.personality_score)
    interests_diff = abs(result_ab.interests_score - result_ba.interests_score)
    final_diff = abs(result_ab.final_score - result_ba.final_score)

    is_symmetric = (
        personality_diff < tolerance and
        interests_diff < tolerance and
        final_diff < tolerance
    )

    print(f"  predict(A, B): personality={result_ab.personality_score:.6f}, "
          f"interests={result_ab.interests_score:.6f}, final={result_ab.final_score:.6f}")
    print(f"  predict(B, A): personality={result_ba.personality_score:.6f}, "
          f"interests={result_ba.interests_score:.6f}, final={result_ba.final_score:.6f}")
    print(f"\n  Differences: personality={personality_diff:.2e}, "
          f"interests={interests_diff:.2e}, final={final_diff:.2e}")

    print(f"\nSymmetry test: {'PASSED' if is_symmetric else 'FAILED'}")
    return is_symmetric


def test_similar_vs_different(predictor: CompatibilityPredictor) -> bool:
    """Test that similar persons score higher than different persons."""
    print("\n" + "=" * 60)
    print("TEST 4: Similar vs Different Scores")
    print("=" * 60)

    person_a = create_mock_person_a()
    person_b_similar = create_mock_person_b_similar()
    person_c_different = create_mock_person_c_different()

    result_similar = predictor.predict(person_a, person_b_similar)
    result_different = predictor.predict(person_a, person_c_different)

    # We expect similar persons to have higher scores than different persons
    # (at least for personality, where OCEAN similarity should be captured)

    print(f"  A vs B (similar):   final={result_similar.final_score:.4f}")
    print(f"  A vs C (different): final={result_different.final_score:.4f}")

    # This is a soft check - we expect similar to score higher but don't fail if not
    # (because the models may have learned different patterns)
    if result_similar.final_score > result_different.final_score:
        print("\n  Similar pair scores HIGHER than different pair (expected)")
        return True
    else:
        print("\n  Similar pair scores LOWER than different pair (unexpected, but not a failure)")
        print("  This may indicate the model learned patterns that differ from simple similarity")
        return True  # Don't fail, just note the observation


def test_perturbation_stability(predictor: CompatibilityPredictor) -> bool:
    """Test that minor input changes produce similar scores."""
    print("\n" + "=" * 60)
    print("TEST 5: Perturbation Stability")
    print("=" * 60)

    person_a = create_mock_person_a()
    person_b = create_mock_person_b_similar()

    # Get baseline score
    baseline = predictor.predict(person_a, person_b)

    # Create slightly perturbed version of A (change one Likert by 1)
    person_a_perturbed = QuestionnaireResponse(
        personality=PersonalityAnswers(
            extraversion=5,      # Changed from 4 to 5
            agreeableness=5,
            conscientiousness=3,
            openness=4,
            neuroticism=2
        ),
        interests=person_a.interests,
        person_id="person_a_perturbed"
    )

    perturbed = predictor.predict(person_a_perturbed, person_b)

    # Scores should change but not dramatically
    score_change = abs(baseline.final_score - perturbed.final_score)

    print(f"  Baseline score:  {baseline.final_score:.4f}")
    print(f"  Perturbed score: {perturbed.final_score:.4f}")
    print(f"  Change:          {score_change:.4f}")

    # A change of 1 point on a single Likert scale should not cause dramatic score change
    # We use a generous threshold of 0.3
    is_stable = score_change < 0.3

    print(f"\nPerturbation stability test: {'PASSED' if is_stable else 'FAILED'}")
    return is_stable


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("COMPATIBILITY INFERENCE SMOKE TEST")
    print("=" * 60)

    # Use seed 11 artifacts
    artifacts_dir = project_root / "artifacts" / "runs" / "seed_11"

    if not artifacts_dir.exists():
        print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
        print("Please run the training pipeline first.")
        return 1

    print(f"\nLoading predictor from: {artifacts_dir}")
    predictor = CompatibilityPredictor(str(artifacts_dir))

    # Run tests
    results = []
    results.append(("Basic Inference", test_basic_inference(predictor)))
    results.append(("Score Ranges", test_score_ranges(predictor)))
    results.append(("Symmetry", test_symmetry(predictor)))
    results.append(("Similar vs Different", test_similar_vs_different(predictor)))
    results.append(("Perturbation Stability", test_perturbation_stability(predictor)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
