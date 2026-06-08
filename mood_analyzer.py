# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Improvements over the starter:
          - Normalizes common contractions so "don't" → "dont" (kept as a negation word)
          - Maps text emoticons ":)" → "happy_emoji" and ":(" → "sad_emoji"
          - Inserts spaces around non-ASCII characters so emoji like 💪 become
            separate tokens instead of being glued to adjacent words
          - Strips ASCII punctuation (commas, periods, apostrophes, etc.)
        """
        cleaned = text.strip().lower()

        # Normalize contractions so negation words survive punctuation stripping
        for contraction, replacement in [
            ("don't", "dont"), ("doesn't", "doesnt"),
            ("isn't", "isnt"), ("wasn't", "wasnt"),
            ("can't", "cant"), ("won't", "wont"),
        ]:
            cleaned = cleaned.replace(contraction, replacement)

        # Map text emoticons to word tokens before we drop punctuation
        cleaned = cleaned.replace(":-)", " happy_emoji ").replace(":)", " happy_emoji ")
        cleaned = cleaned.replace(":-(", " sad_emoji ").replace(":(", " sad_emoji ")

        # Build the cleaned string character-by-character so we can:
        #   - Keep alphanumerics and underscores unchanged
        #   - Keep non-ASCII characters (emoji) but surround them with spaces
        #     so they tokenize as their own tokens
        #   - Replace all ASCII punctuation with a space
        result: List[str] = []
        for char in cleaned:
            if ord(char) > 127:
                result.append(f" {char} ")
            elif char.isalnum() or char == "_" or char == " ":
                result.append(char)
            else:
                result.append(" ")

        tokens = [t for t in "".join(result).split() if t]
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private helper: single pass that collects score + hit lists
    # ------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Core analysis pass.  Returns (score, positive_hits, negative_hits).

        Modeling improvements implemented here:
          1. Negation handling — words after "not", "never", "no", "dont",
             etc. have their sentiment flipped.  "not happy" becomes a
             negative signal; "not bad" becomes a positive signal.
          2. Emoji signals — emoji characters registered in POSITIVE_WORDS
             or NEGATIVE_WORDS (e.g. 💪, 🙄, 😤) are scored the same way
             as regular words after preprocess() splits them into tokens.
        """
        tokens = self.preprocess(text)

        # Words that flip the sentiment of the immediately following word
        negators = {
            "not", "never", "no", "dont", "doesnt",
            "isnt", "wasnt", "cant", "wont", "barely", "hardly",
        }

        score = 0
        pos_hits: List[str] = []
        neg_hits: List[str] = []
        negate_next = False

        for token in tokens:
            if token in negators:
                negate_next = True
                continue

            mult = -1 if negate_next else 1
            negate_next = False  # negation consumed by this token

            if token in self.positive_words:
                score += mult
                if mult > 0:
                    pos_hits.append(token)
                else:
                    neg_hits.append(f"not {token}")  # negated positive → negative signal
            elif token in self.negative_words:
                score -= mult
                if mult > 0:
                    neg_hits.append(token)
                else:
                    pos_hits.append(f"not {token}")  # negated negative → positive signal

        return score, pos_hits, neg_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score; negative words decrease it.
        Negation words ("not", "never", "no", …) flip the sign of the
        immediately following sentiment word.
        """
        score, _, _ = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the analysis for a piece of text into a mood label.

        Label logic:
          - Both positive AND negative signals present → "mixed"
          - Only positive signals (score > 0)          → "positive"
          - Only negative signals (score < 0)          → "negative"
          - No signals at all (score == 0)             → "neutral"

        The "mixed" category fires whenever there is evidence of both
        positive and negative tone, regardless of which side wins the
        overall score.
        """
        score, pos_hits, neg_hits = self._analyze(text)
        if pos_hits and neg_hits:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example output:
          'Score = -1 | positive: [] | negative: ["not excited"]'
        """
        score, pos_hits, neg_hits = self._analyze(text)
        return (
            f"Score = {score} "
            f"| positive: {pos_hits or []} "
            f"| negative: {neg_hits or []}"
        )
