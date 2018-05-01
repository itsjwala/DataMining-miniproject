from flask import Flask, request, jsonify
import pickles_algos
from algos import *

app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file("index.html")


@app.route('/predict')
def prediction():
    X_test_input = []
    age = request.values['age']
    X_test_input.append(age)
    salary = request.values['salary']
    X_test_input.append(salary)
    algo = request.values['algo']
    data = dict()
    if algo == 'randForest':
        data['prediction'] = randforest(X_test_input)
    elif algo == 'naiveBayes':
        data['prediction'] = naivebayes(X_test_input)
    elif algo == 'knn':
        data['prediction'] = k_nearest_neighbours(X_test_input)
    elif algo == 'decisionTree':
        data['prediction'] = decision_tree(X_test_input)
    else:
        data['prediction'] = 'None'

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
