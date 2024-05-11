from flask import Flask, render_template, request
from ml_logic import calculate_weighted_sentiment_score, assess_mental_state
import os

app = Flask(__name__)

# Set the template folder explicitly
template_dir = os.path.abspath('templates')
app.template_folder = template_dir

# Define routes and functions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_mental_state', methods=['POST'])
def calculate_mental_state():
    if request.method == 'POST':
        # Extract data from the form
        responses = request.form.to_dict()
        
        # Calculate weighted sentiment score
        weighted_sentiment_score = calculate_weighted_sentiment_score(responses)
        
        # Assess mental state based on the weighted sentiment score
        mental_state = assess_mental_state(weighted_sentiment_score)
        
        # Render the result template with the mental state
        return render_template('result.html', mental_state=mental_state)

if __name__ == "__main__":
    app.run(debug=True)
