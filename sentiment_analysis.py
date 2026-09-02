import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

reviews = [
    "The build quality is exceptional! Outstanding product.",
    "Completely broke after two days. Horrible quality, do not buy.",
    "It works okay, nothing special but serves its purpose.",
    "Decent value for money. Arrival was fast and well-packaged.",
    "Terrible customer service. Very disappointed with my purchase.",
    "Absolutely love it! Exceeded my expectations in every way.",
    "Average item. A bit overpriced for what it offers."
]


def analyze_sentiment():
    sia = SentimentIntensityAnalyzer()
    results = []

    for text in reviews:
        scores = sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "Review": text,
            "Compound_Score": compound,
            "Sentiment": label
        })

    df = pd.DataFrame(results)
    df.to_csv("sentiment_summary.csv", index=False)

    # Plotting results
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Sentiment", palette="coolwarm", order=["Positive", "Neutral", "Negative"])
    plt.title("Product Review Sentiment Distribution", fontweight="bold")
    plt.ylabel("Number of Reviews")
    plt.savefig("sentiment_distribution.png", dpi=300)

    print("Analysis finished. Summary CSV and chart saved successfully.")


if __name__ == "__main__":
    analyze_sentiment()