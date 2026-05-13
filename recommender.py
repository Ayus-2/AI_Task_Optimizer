def recommend_task(emotion):

    if emotion == "Happy":
        return "Assign creative or challenging tasks"

    elif emotion == "Stressed":
        return "Assign light workload or allow short break"

    else:
        return "Assign regular routine tasks"