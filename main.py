from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

#load a model
model=joblib.load('hiringmodel1.pkl')


@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/predict', methods=['post'])
def predict():
    exp=request.form.get('experience')
    score=request.form.get('test_score')
    interview_score=request.form.get('interview_score')

    print(exp,score,interview_score,end=' \t')

    prediction= model.predict([[int(exp),int(score),int(interview_score)]])
    output=round(prediction[0],2)

    return render_template('base.html',prediction_text=f'employee salary will be $ {output}')

if __name__=='__main__': # this is  a mand.syntax it is independent of the file main.
    app.run(debug=True)


