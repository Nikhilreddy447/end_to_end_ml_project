from flask import Flask, render_template, request

app = Flask(__name__)

# Import your machine learning model
from model import ml_model

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input1 = float(request.form['number1'])
        input2 = float(request.form['number2'])
        input3 = float(request.form['number3'])
        input4 = float(request.form['number4'])
        
        # Make predictions using your model
        output = ml_model(input1, input2, input3, input4)

        return render_template('index.html', output=output)
    
    return render_template('index.html', output=None)

if __name__ == '__main__':
    app.run(debug=True)
