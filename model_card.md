# Model Card: Mood Machine

This model card covers both versions of the Mood Machine classifier:

1. A **rule-based model** in `mood_analyzer.py`
2. A **machine learning model** in `ml_experiments.py` (scikit-learn logistic regression)

---

## 1. Model Overview

**Model type:** Both models were built and compared.

**Intended purpose:** Classify short social-media-style text messages (under ~15 words) into one of four mood categories: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

*Rule-based:* Text is preprocessed (lowercased, punctuation stripped, text emoticons converted, emoji characters split into their own tokens). Each token is checked against a `POSITIVE_WORDS` and `NEGATIVE_WORDS` list. Negation words ("not", "never", "no", "dont", …) flip the sentiment of the immediately following word. If both positive and negative signals are found the label is `mixed`; otherwise the sign of the total score determines the label.

*ML:* Posts are converted into bag-of-words count vectors (`CountVectorizer`) and a logistic regression classifier is fitted on those vectors and the hand-assigned labels.

---

## 2. Data

**Dataset description:** 14 short posts total — 6 from the starter kit and 8 added during Part 1 of this lab. All posts are under 15 words and written to resemble real social media messages.

**Labeling process:** Labels were assigned by reading each post and judging the dominant emotional tone. Several required judgment calls:

- `"I absolutely love being stuck in traffic 🙄"` — labeled `negative` because the context is sarcastic; only the 🙄 emoji signals the true frustration. Another reader might call it `mixed`.
- `"this movie was so mid"` — labeled `negative` because "mid" is Gen Z slang for mediocre/bad, but without that cultural context the label looks arbitrary.
- `"feeling lowkey anxious but hyped for tomorrow"` — labeled `mixed`; fear and anticipation are both genuine, and either could be the dominant reading.

**Dataset characteristics:**

- Contains Gen Z slang: "mid", "lowkey", "hyped", "vibes", "immaculate", "crushing it"
- Contains emoji signals: 💪 (strength/pride), 🙄 (eye-roll/sarcasm), 😤 (frustration)
- One sarcastic post that contradicts keyword-level tone
- Three posts expressing mixed feelings
- All examples are short and written in North American English

**Possible issues:**

- Only 14 examples — far too small for any model to generalize
- No held-out test set; every metric is training accuracy
- Labels reflect one person's interpretation; inter-annotator agreement was not measured
- Slang and cultural references skew toward a specific demographic

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

1. **Preprocessing:** lowercase → normalize contractions (e.g. "don't" → "dont") → replace text emoticons (":)" → `happy_emoji`) → insert spaces around non-ASCII emoji chars → strip ASCII punctuation.
2. **Negation handling:** words after "not", "never", "no", "dont", "cant", "wont", etc. have their sentiment sign flipped. `"not happy"` scores as −1 (negative signal). `"not bad"` scores as +1 (positive signal).
3. **Scoring:** each token adds or subtracts 1 based on membership in `POSITIVE_WORDS` / `NEGATIVE_WORDS`. Negation multiplies by −1 before adding.
4. **Label mapping:** if both positive and negative hits exist → `"mixed"`; else score > 0 → `"positive"`, score < 0 → `"negative"`, score == 0 → `"neutral"`.

Word lists were expanded during the lab to include slang (`hyped`, `vibes`, `immaculate`, `crushing`, `decent`, `hopeful`) and emoji characters (`💪`, `🙄`, `😤`).

**Strengths:**

- Fully transparent — every decision traces back to a named word in the lists
- Negation works for simple adjacent patterns ("not happy", "not excited")
- Emoji characters in `POSITIVE_WORDS`/`NEGATIVE_WORDS` are handled automatically after preprocessing
- Zero training data required

**Weaknesses:**

- Cannot detect sarcasm — "I absolutely love being stuck in traffic" reads "love" as a genuine positive even with the 🙄 partially correcting it
- Unknown slang ("mid") is invisible; the model defaults to `neutral`
- Negation scope is one token only — "not really that happy" would negate "really", not "happy"
- Any post dominated by a single strong keyword will be over-determined by that one word

---

## 4. How the ML Model Works

**Features used:** Bag of words via `CountVectorizer`. Each post is represented as a vector of word counts across the full vocabulary of `SAMPLE_POSTS`.

**Training data:** The model trained on all 14 `SAMPLE_POSTS` with their `TRUE_LABELS` — the same data it was evaluated on.

**Training behavior:** With only 14 examples the logistic regression model memorized the entire training set and reported **100% training accuracy**. Changing even one label caused the model to shift predictions on unrelated examples, which shows how tightly it depends on the specific examples it was trained on.

**Strengths and weaknesses:**

*Strengths:* The ML model learned that "mid" maps to `negative` and that the sarcastic "love" post is `negative` — both of which the rule-based model got wrong — because it saw those labels during training.

*Weaknesses:* That knowledge is pure memorization. On a new post containing "mid" that it has never seen, the ML model would ignore the word just as the rule-based model does. It has not learned anything general about sarcasm or slang; it has just matched exact training examples.

---

## 5. Evaluation

**How evaluated:** Both models ran against the full 14-post labeled dataset. The rule-based model reports the accuracy printed in `main.py`; the ML model uses `sklearn.metrics.accuracy_score` in `ml_experiments.py`.

| Model | Accuracy on SAMPLE_POSTS |
|---|---|
| Rule-based | **0.86** (12/14) |
| ML (logistic regression) | **1.00** (14/14) — training accuracy |

**Examples of correct rule-based predictions:**

- `"Today was a terrible day"` → **negative** ✓  
  "terrible" is in `NEGATIVE_WORDS`, no countering signals. Score = −1.

- `"I am not happy about this"` → **negative** ✓  
  Negation handling: "not" flips "happy" → neg_hits = ["not happy"], score = −1.

- `"exhausted but crushing it 💪"` → **mixed** ✓  
  "exhausted" hits `NEGATIVE_WORDS`; "crushing" and "💪" both hit `POSITIVE_WORDS`. Both lists non-empty → mixed.

**Examples of incorrect rule-based predictions:**

- `"I absolutely love being stuck in traffic 🙄"` → predicted **mixed**, true **negative**  
  The model reads "love" as a genuine positive signal (score +1) and 🙄 as negative (score −1). With both signals it returns `mixed`. It cannot recognize that the sarcasm makes "love" false.

- `"this movie was so mid"` → predicted **neutral**, true **negative**  
  "mid" (Gen Z slang for mediocre/bad) is absent from all word lists. Score = 0, no signals → `neutral`. This is purely a vocabulary gap.

The ML model corrected both of these on the training set, but only because it memorized the exact posts. The underlying reasoning gap remains.

---

## 6. Limitations

1. **Tiny dataset.** 14 examples cannot represent the diversity of real language. The ML model overfits completely; the rule-based model can only detect words it has seen.
2. **No sarcasm detection.** Both models rely on keyword polarity. Sarcasm inverts the relationship between keywords and meaning — a fundamental problem that keywords alone cannot fix.
3. **Unknown slang.** Words like "mid", "no cap", "bussin", or "it's giving" carry clear sentiment in Gen Z English but score as neutral unless explicitly added to the word lists. Word lists require constant manual maintenance.
4. **Single-token negation scope.** "Not really that happy" only negates "really"; "happy" still scores as positive. Multi-word negation spans need a different approach.
5. **English-only, North American skew.** The dataset contains no non-English text, AAVE, British English, or other dialect variations.
6. **No real test set.** All accuracy numbers are training accuracy. Generalization performance is unknown.

---

## 7. Ethical Considerations

- **Mental health risk.** A message like `"not excited, just tired of everything"` could signal burnout or depression. A false `neutral` or `positive` prediction — especially in a context like mental health monitoring — could cause real harm by masking distress.
- **Slang and dialect gaps.** The system is optimized for a narrow slice of English. Users from other communities may be systematically misclassified. For example, AAVE uses "bad" and "sick" in ways that directly contradict their polarity in the word lists.
- **Sarcasm and trust.** A user who writes sarcastically and gets a confident `positive` prediction may lose trust in the system — or worse, the system may be used to make decisions about them based on the wrong label.
- **Privacy.** Even a toy model that analyzes personal messages requires informed consent. Deploying this system on real user data without disclosure violates privacy norms.
- **Over-reliance.** With 86% accuracy on 14 in-domain training examples, the real-world accuracy could be far lower. Any production use should treat predictions as weak hints, not ground truth, and include human review for consequential decisions.

---

## 8. Ideas for Improvement

- **More labeled data.** Even 500 diverse, carefully labeled examples would substantially improve the ML model and reveal which rule-based gaps matter most.
- **Real train/test split.** Split the dataset before evaluation so accuracy reflects generalization, not memorization.
- **TF-IDF instead of CountVectorizer.** Down-weighting common words like "the", "is", and "so" would reduce noise and improve the bag-of-words features.
- **Wider negation scope.** Extend negation to cover a window of 2–3 tokens ("not really that happy") instead of only the next token.
- **Slang dictionary.** Import a community-maintained lexicon (e.g., Urban Dictionary API or a labeled slang dataset) to handle words like "mid", "cap", "bussin", "lowkey".
- **Sentence embeddings.** Use a pre-trained model (e.g., `sentence-transformers`) to represent posts as semantic vectors rather than exact keyword counts. This would allow the model to generalize to unseen phrasing.
- **Sarcasm detection.** Requires pragmatic context — a fine-tuned transformer (BERT, RoBERTa) trained on sarcasm-annotated data is the most realistic path to handling this reliably.
- **Emoji lexicon.** Rather than manually adding emoji to `POSITIVE_WORDS`/`NEGATIVE_WORDS`, use a published emoji sentiment lexicon (e.g., the Emoji Sentiment Ranking) to cover all common emoji.
