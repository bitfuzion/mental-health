import nltk
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Define a dictionary to store questions and their respective marks
question_weights = {
    # Emotional Well-being (12 questions)
    "Easy to sleep or do you observe change in sleeping patterns?": 5,
    "Worry or stress about various phases of life?": 5,
    "Feel a sense of worthlessness or hopelessness?": 5,
    "Feel misunderstood or unsupported from friends, teachers or family members?": 5,
    "Do you experience any recurring thoughts or memories that are upsetting or difficult to control?": 5,
    "Feel a sense of emptiness or numbness?": 5,
    "Experienced feel of guilt or shame related to any event for a longer time?": 5,
    "Do you find it difficult to control your emotions in stressful situations?": 5,
    "Do you feel physically tired or drained most of the time, even after getting enough sleep?": 5,
    "Do you find yourself easily irritated or frustrated by minor things?": 5,
    "Do you feel generally optimistic and hopeful about the future?": 5,
    "In the past week, have you had any thoughts of hurting yourself or others?": 5,

    # Social Support (9 questions)
    "Experience easy relationship with friends, teachers and family members?": 4,
    "Do you enjoy spending time with friends and family members or participating in activities you find fun?": 4,
    "Do you find it helpful to express your feelings by talking to someone you trust?": 4,
    "I trust that if I confide in others, they will be supportive?": 4,
    "I actively keep in touch with friends and family?": 4,
    "When things are tough, I reach out for support?": 4,
    "Have difficulty in maintaining or forming relationships?": 4,
    "Do you require excessive admiration from others?": 4,
    "I have strong relationships with people I care about?": 4,

    # Stress and Coping (9 questions)
    "Find difficulty concentrating on tasks or completing assignments?": 4,
    "Find hard to control my emotions and may react impulsively?": 4,
    "Easily manage all the work and handles the frustration from the work?": 4,
    "Do you engage in rituals that provide temporary relief to your anxiety, such as counting, checking, or cleaning?": 4,
    "Are you preoccupied with fantasies of unlimited success, power, brilliance, beauty, or ideal love?": 4,
    "Do you dread going to work or coming back from vacation?": 4,
    "Trouble falling or staying asleep, or sleeping too much?": 4,
    "Do you feel safe at school and at home?": 4,
    "Poor appetite or overeating?": 4,

    # Self-esteem and Sense of Worth (8 questions)
    "Experience feel of guilt or shame related to any event for a longer time?": 3,
    "I feel that life is very rewarding?": 3,
    "I feel grateful for what I have?": 3,
    "I am able to find the goodness in myself and others?": 3,
    "I have a sense of meaning and purpose in my life?": 3,
    "I feel good about the choices I've made in my life?": 3,
    "I am able to identify and express my emotions?": 3,
    "I feel like I am not living up to my own expectations, or those of others?": 3,

    # Decision Making and Cognitive Functioning (4 questions)
    "Do you worry or stress about various phases of life?": 3,
    "Do you find it difficult to make decisions, feeling indecisive or confused?": 3,
    "When I experience a strong emotion, I usually know why it's hitting me?": 3,
    "I procrastinate and/or avoid dealing with important things in my life?": 3,

    # Adaptability and Resilience (5 questions)
    "Easily adapt the changing new environment?": 3,
    "Do you feel dissatisfied or disillusioned with your work or other responsibilities?": 3,
    "I'm able to bounce back from setbacks?": 3,
    "I manage my time and my obligations; most days life feels under control?": 3,
    "Are you experiencing fast, uncontrollable mood changes which tend to go extreme levels?": 3
}

def calculate_weighted_sentiment_score(responses):
    weighted_sum = 0
    for question, response in responses.items():
        sentiment_score = sid.polarity_scores(response)['compound']
        weighted_score = sentiment_score * question_weights[question]
        weighted_sum += weighted_score
    return weighted_sum

def assess_mental_state(weighted_sentiment_score):
    # Define thresholds for different mental states
    very_happy_threshold = 7
    happy_threshold = 4
    normal_threshold = 0
    stressed_threshold = -4
    very_stressed_threshold = -7

    # Determine the mental state based on the weighted sentiment score
    if weighted_sentiment_score >= very_happy_threshold:
        return "Very Happy"
    elif weighted_sentiment_score >= happy_threshold:
        return "Happy"
    elif weighted_sentiment_score >= normal_threshold:
        return "Normal"
    elif weighted_sentiment_score >= stressed_threshold:
        return "Stressed"
    elif weighted_sentiment_score >= very_stressed_threshold:
        return "Very Stressed"
    else:
        return "Extremely Stressed"