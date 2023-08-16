from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Download VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

def sa(text):
	# Initialize the SentimentIntensityAnalyzer
	sia = SentimentIntensityAnalyzer()

	# Perform sentiment analysis
	sentiment_scores = sia.polarity_scores(text)

	# Interpret sentiment scores
	if sentiment_scores['compound'] >= 0.05:
	    sentiment = 'Positive'
	elif sentiment_scores['compound'] <= -0.05:
	    sentiment = 'Negative'
	else:
	    sentiment = 'Neutral'
	return sentiment_scores,sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    result = sa(user_input)
    return render_template('index.html', user_input=result[0], processed_output=result[1])

if __name__ == '__main__':
    app.run(debug=True)
    
    
    





