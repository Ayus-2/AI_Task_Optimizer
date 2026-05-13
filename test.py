from emotion import detect_emotion
from recommender import recommend_task

text = input("Enter employee text: ")

emotion = detect_emotion(text)

task = recommend_task(emotion)

print("Detected Emotion:", emotion)
print("Recommended Task:", task)