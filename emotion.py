from textblob import TextBlob

def detect_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Happy"
    elif polarity < 0:
        return "Stressed"
    else:
        return "Neutral"