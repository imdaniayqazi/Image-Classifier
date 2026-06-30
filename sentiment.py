from textblob import TextBlob

print("=" * 45)
print("   SENTIMENT ANALYZER — by Dania Yasir")
print("=" * 45)
print("Type any text and I'll analyze its mood!")
print("Type 'quit' to exit\n")

while True:
    text = input("Enter text: ")
    
    if text.lower() == "quit":
        print("Goodbye!")
        break
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        mood = "😊 POSITIVE"
        color = "Great vibes!"
    elif polarity < -0.1:
        mood = "😞 NEGATIVE"
        color = "Not so good!"
    else:
        mood = "😐 NEUTRAL"
        color = "Pretty balanced!"
    
    print("\n── ANALYSIS RESULT ──")
    print(f"  Mood       : {mood}")
    print(f"  Polarity   : {polarity:.2f}  (-1 negative → +1 positive)")
    print(f"  Subjectivity: {subjectivity:.2f}  (0 factual → 1 opinion)")
    print(f"  Verdict    : {color}")
    print("─" * 25 + "\n")