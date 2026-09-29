"""
CodeAlpha - Data Analytics Internship
TASK 4: Sentiment Analysis

- Classifies text as Positive / Negative / Neutral using VADER (lexicon-based NLP)
- Detects specific emotions using a keyword emotion lexicon
- Works on product reviews, tweets, news headlines, etc.
- Visualises patterns and prints business insights

DATA: By default a small built-in sample of product reviews is used so the
script runs immediately. To use your own data (e.g. an Amazon reviews CSV),
set CSV_PATH and TEXT_COLUMN below.

Install:  pip install pandas matplotlib seaborn nltk
Run:      python task4_sentiment_analysis.py
"""

import os
import re
from collections import Counter

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ------------------------------------------------------------------ CONFIG
CSV_PATH = None            # e.g. "reviews.csv"
TEXT_COLUMN = "review"     # column containing the text
OUT_DIR = "sentiment_output"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))

# ------------------------------------------------------------------ DATA
SAMPLE = [
    ("Amazing phone! The battery lasts all day and the camera is fantastic.", "Phone"),
    ("Terrible quality. It stopped working after two days. Totally disappointed.", "Phone"),
    ("It's okay, does the job. Nothing special.", "Phone"),
    ("Absolutely love these headphones, the sound is crystal clear!", "Headphones"),
    ("Worst purchase ever. The left earbud broke within a week. Very angry.", "Headphones"),
    ("Delivery was fast and the packaging was fine.", "Headphones"),
    ("The laptop is super fast and the screen is gorgeous. Highly recommend!", "Laptop"),
    ("Overpriced and slow. I regret buying it.", "Laptop"),
    ("Battery is average, keyboard is comfortable. Fair for the price.", "Laptop"),
    ("Customer support was rude and never solved my problem. Awful experience.", "Service"),
    ("Support team was helpful and friendly, they fixed everything quickly. Thank you!", "Service"),
    ("I am scared this product is unsafe, it overheats badly.", "Phone"),
    ("Great value for money, I'm very happy with this purchase.", "Headphones"),
    ("Not worth it. Cheap plastic, feels fragile and useless.", "Laptop"),
    ("Received the item on Tuesday. Box was standard size.", "Service"),
    ("What a pleasant surprise! Exceeded my expectations, brilliant product.", "Phone"),
    ("The screen cracked easily and the seller ignored me. Frustrating and sad.", "Laptop"),
    ("Good sound, decent build, would buy again.", "Headphones"),
    ("Horrible. Disgusting smell from the charger and it is a waste of money.", "Phone"),
    ("Works as described.", "Service"),
]

if CSV_PATH:
    df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COLUMN])
    df = df.rename(columns={TEXT_COLUMN: "review"})
    if "category" not in df.columns:
        df["category"] = "All"
else:
    df = pd.DataFrame(SAMPLE, columns=["review", "category"])

print(f"Loaded {len(df)} texts.\n")


# ------------------------------------------------------------------ CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+|#", "", text)                  # mentions / hashtags
    text = re.sub(r"[^a-z\s!?']", " ", text)            # keep letters (and ! ? ' for VADER cues)
    return re.sub(r"\s+", " ", text).strip()


df["clean"] = df["review"].apply(clean_text)

# ------------------------------------------------------------------ SENTIMENT (VADER)
sia = SentimentIntensityAnalyzer()


def classify(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


scores = df["review"].apply(sia.polarity_scores).apply(pd.Series)
df = pd.concat([df, scores], axis=1)
df["sentiment"] = df["compound"].apply(classify)

# ------------------------------------------------------------------ EMOTION LEXICON
EMOTION_LEXICON = {
    "joy":      {"love", "happy", "great", "amazing", "fantastic", "gorgeous", "brilliant", "pleasant",
                 "recommend", "excellent", "wonderful", "perfect", "thank", "thanks", "delighted"},
    "anger":    {"angry", "rude", "worst", "awful", "horrible", "terrible", "furious", "hate", "ignored",
                 "disgusting", "frustrating", "unacceptable"},
    "sadness":  {"sad", "disappointed", "regret", "unfortunately", "useless", "broke", "broken", "cracked"},
    "fear":     {"scared", "unsafe", "dangerous", "overheats", "afraid", "worried", "risk", "fragile"},
    "trust":    {"reliable", "helpful", "friendly", "quickly", "described", "recommend", "value", "fixed"},
    "surprise": {"surprise", "surprised", "unexpected", "exceeded", "wow", "shocked"},
}


def detect_emotions(text):
    words = set(re.findall(r"[a-z']+", text))
    found = [emo for emo, lex in EMOTION_LEXICON.items() if words & lex]
    return found or ["none"]


df["emotions"] = df["clean"].apply(detect_emotions)

# ------------------------------------------------------------------ RESULTS
print("SENTIMENT DISTRIBUTION")
dist = df["sentiment"].value_counts()
print(dist.to_string(), "\n")

print("SAMPLE RESULTS")
print(df[["review", "compound", "sentiment", "emotions"]].head(10).to_string(index=False), "\n")

emotion_counts = Counter(e for lst in df["emotions"] for e in lst if e != "none")
print("EMOTION COUNTS:", dict(emotion_counts), "\n")


def top_words(sentiment, n=8):
    words = " ".join(df.loc[df.sentiment == sentiment, "clean"]).split()
    words = [w for w in words if w not in STOP and len(w) > 2 and w not in {"!", "?"}]
    return Counter(w.strip("!?'") for w in words).most_common(n)


for s in ["Positive", "Negative"]:
    print(f"Top words in {s} texts:", top_words(s))

by_cat = df.groupby("category")["compound"].mean().sort_values()
print("\nAVERAGE SENTIMENT SCORE BY CATEGORY\n", by_cat.round(3).to_string())

# ------------------------------------------------------------------ VISUALS
palette = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].pie(dist.values, labels=dist.index, autopct="%1.0f%%", startangle=90,
               colors=[palette[k] for k in dist.index], wedgeprops={"edgecolor": "white"})
axes[0, 0].set_title("Overall Sentiment Share")

sns.histplot(df["compound"], bins=15, kde=True, color="#3498db", ax=axes[0, 1])
axes[0, 1].axvline(0, color="black", ls="--")
axes[0, 1].set(title="Distribution of Compound Scores (-1 to +1)", xlabel="Compound score")

ct = pd.crosstab(df["category"], df["sentiment"])
ct = ct.reindex(columns=[c for c in ["Positive", "Neutral", "Negative"] if c in ct.columns])
ct.plot(kind="bar", stacked=True, ax=axes[1, 0], color=[palette[c] for c in ct.columns])
axes[1, 0].set(title="Sentiment by Product Category", xlabel="", ylabel="Count")
axes[1, 0].tick_params(axis="x", rotation=0)

if emotion_counts:
    emo = pd.Series(emotion_counts).sort_values()
    emo.plot(kind="barh", ax=axes[1, 1], color="#9b59b6")
axes[1, 1].set(title="Detected Emotions", xlabel="Number of texts")

plt.suptitle("Sentiment Analysis Report", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sentiment_report.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------ SAVE
out = df.drop(columns=["clean"]).copy()
out["emotions"] = out["emotions"].apply(", ".join)
out.to_csv(os.path.join(OUT_DIR, "sentiment_results.csv"), index=False)

# ------------------------------------------------------------------ INSIGHTS
pos_pct = (df.sentiment == "Positive").mean()
neg_pct = (df.sentiment == "Negative").mean()
print("\n" + "=" * 60 + "\nBUSINESS INSIGHTS\n" + "=" * 60)
print(f" * {pos_pct:.0%} of feedback is positive, {neg_pct:.0%} is negative.")
print(f" * Best-rated category : {by_cat.index[-1]} ({by_cat.iloc[-1]:.2f})")
print(f" * Weakest category    : {by_cat.index[0]} ({by_cat.iloc[0]:.2f})")
print(" * Anger/sadness/fear words point to quality defects, poor support and safety concerns.")
print(" * Marketing: reuse the praised features (battery, sound, screen) in campaigns.")
print(" * Product: investigate durability and overheating complaints first.")
print(f"\nSaved: {OUT_DIR}/sentiment_report.png and {OUT_DIR}/sentiment_results.csv")
