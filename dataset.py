"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # added during lab expansion
    "hopeful",
    "decent",
    "hyped",
    "immaculate",
    "crushing",
    "💪",
    "vibes",
    "proud",
    "wonderful",
    "happy_emoji",  # mapped from :) in preprocess
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # added during lab expansion
    "anxious",
    "exhausted",
    "worst",
    "🙄",
    "😤",
    "miserable",
    "sad_emoji",    # mapped from :( in preprocess
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- added during lab (Part 1) ---
    "not bad at all actually pretty decent",        # negation → positive resolution
    "I absolutely love being stuck in traffic 🙄",  # sarcasm: rule model predicts mixed, not negative
    "feeling lowkey anxious but hyped for tomorrow", # slang + mixed emotions
    "this movie was so mid",                        # Gen Z slang; "mid" not in word lists
    "exhausted but crushing it 💪",                 # mixed: fatigue vs. pride
    "worst day ever 😤",                            # clear negative with emoji signal
    "vibes are immaculate today",                   # positive slang
    "not excited about this at all",               # negation → negative result
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # --- added during lab ---
    "positive",  # "not bad at all actually pretty decent"
    "negative",  # "I absolutely love being stuck in traffic 🙄" (sarcasm)
    "mixed",     # "feeling lowkey anxious but hyped for tomorrow"
    "negative",  # "this movie was so mid" (Gen Z slang for mediocre/bad)
    "mixed",     # "exhausted but crushing it 💪"
    "negative",  # "worst day ever 😤"
    "positive",  # "vibes are immaculate today"
    "negative",  # "not excited about this at all"
]

assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS has {len(SAMPLE_POSTS)} entries but TRUE_LABELS has {len(TRUE_LABELS)}. "
    "Add a matching label for every post."
)
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
