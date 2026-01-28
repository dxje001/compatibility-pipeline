# UI Smoke Test Manual

Manual test scenarios for the Compatibility Scoring UI.

## Prerequisites

1. Training artifacts exist at `artifacts/runs/seed_11/`
2. Streamlit installed: `pip install streamlit`
3. Start UI: `streamlit run ui/app.py`

---

## Scenario 1: High Compatibility Pair

**Goal:** Verify similar profiles produce high compatibility scores.

### Person A Inputs

**Personality:**
- Extraversion: 4 (Agree)
- Agreeableness: 5 (Strongly Agree)
- Conscientiousness: 3 (Neutral)
- Openness: 4 (Agree)
- Neuroticism: 2 (Disagree)

**Interests:**
- Religion: Agnosticism
- About Me: "I love hiking in the mountains, reading science fiction novels, and trying new restaurants around the city. I work in the tech industry as a software developer and enjoy learning new programming languages in my spare time. Looking for someone who shares my curiosity about the world and enjoys deep conversations about ideas."
- Lifestyle: Mostly healthy (social drinking only)
- Family: Single, no kids, wants kids
- Education: Master's degree

### Person B Inputs

**Personality:**
- Extraversion: 4 (Agree)
- Agreeableness: 4 (Agree)
- Conscientiousness: 3 (Neutral)
- Openness: 5 (Strongly Agree)
- Neuroticism: 2 (Disagree)

**Interests:**
- Religion: Agnosticism
- About Me: "Passionate about outdoor adventures, books, and exploring new cuisines. I'm a software engineer who loves tinkering with new technologies and building side projects. Seeking a partner who is intellectually curious and values personal growth. I enjoy weekend camping trips and hosting dinner parties with friends."
- Lifestyle: Mostly healthy (social drinking only)
- Family: Single, no kids, wants kids
- Education: Bachelor's degree

### Expected Results

- Personality Score: > 70%
- Interests Score: > 40%
- Final Score: > 55%
- Confidence Level: High
- No validation errors

---

## Scenario 2: Low Compatibility Pair

**Goal:** Verify different profiles produce low compatibility scores.

### Person A Inputs

**Personality:**
- Extraversion: 5 (Strongly Agree)
- Agreeableness: 4 (Agree)
- Conscientiousness: 2 (Disagree)
- Openness: 5 (Strongly Agree)
- Neuroticism: 1 (Strongly Disagree)

**Interests:**
- Religion: Atheism
- About Me: "I'm an adventure seeker who loves spontaneous travel, loud music festivals, and meeting new people every weekend. I work in marketing and thrive in fast-paced, social environments. Looking for someone who can keep up with my energy and doesn't mind last-minute plans."
- Lifestyle: Relaxed (regular drinking/smoking)
- Family: Single, no kids, doesn't want kids
- Education: Bachelor's degree

### Person B Inputs

**Personality:**
- Extraversion: 1 (Strongly Disagree)
- Agreeableness: 2 (Disagree)
- Conscientiousness: 5 (Strongly Agree)
- Openness: 1 (Strongly Disagree)
- Neuroticism: 5 (Strongly Agree)

**Interests:**
- Religion: Christianity (serious)
- About Me: "I prefer quiet evenings at home with a good book or watching documentaries. I'm an accountant who values structure, routine, and careful planning. Faith is very important to me and I attend church regularly. Looking for someone with traditional values who wants to build a stable family life."
- Lifestyle: Very healthy (no drinking, smoking, or drugs)
- Family: Single, has kids
- Education: High school

### Expected Results

- Personality Score: < 40%
- Final Score: < 45%
- Confidence Level: High
- No validation errors

---

## Scenario 3: Validation & Edge Cases

**Goal:** Verify input validation works correctly.

### Test 3.1: Empty About Me

1. Fill all fields for Person A normally
2. Leave Person B's "About Me" field completely empty
3. Click "Compute Compatibility"

**Expected:** Error message "Person B: Please fill in the 'About Me' field"

### Test 3.2: Short About Me (Warning)

1. Fill Person A with full text (200+ chars)
2. Fill Person B with only "I like stuff" (short text)
3. Click "Compute Compatibility"

**Expected:**
- Warning about short text for Person B
- Computation proceeds (not blocked)
- Confidence Level: Low or Medium

### Test 3.3: All "Prefer not to say"

1. Set all categorical fields to "Prefer not to say" for both persons
2. Fill personality normally
3. Fill About Me with 200+ characters

**Expected:**
- Computation succeeds
- Confidence Level: Low
- Confidence note mentions "Limited information provided"

### Test 3.4: Identical Inputs

1. Fill Person A and Person B with exactly the same values
2. Copy-paste the same About Me text

**Expected:**
- Personality Score: Close to 100%
- High overall compatibility
- Demonstrates symmetry

---

## Validation Checklist

| Test | Status | Notes |
|------|--------|-------|
| Scenario 1: High compatibility | [ ] | |
| Scenario 2: Low compatibility | [ ] | |
| Test 3.1: Empty text error | [ ] | |
| Test 3.2: Short text warning | [ ] | |
| Test 3.3: All unknown fields | [ ] | |
| Test 3.4: Identical inputs | [ ] | |
| Score breakdown expands | [ ] | |
| Progress bar matches final score | [ ] | |
| Interpretation text matches score range | [ ] | |

---

## Known Limitations

1. **Interests score variance:** Short free-text inputs produce lower similarity variance due to sparse TF-IDF vectors. Minimum 200 characters recommended.

2. **Categorical coverage:** Some original OkCupid fields (diet, body type, pets, job, sign) are not captured in the UI. These are filled with "unknown" at inference time.

3. **Personality granularity:** The UI asks 1 question per OCEAN dimension. The original model used 10 questions per dimension. Raw features are set to neutral values.
