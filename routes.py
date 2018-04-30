from flask import Flask, request, jsonify
import pickles_algos
from algos import *

app = Flask(__name__)


@app.route('/')
def root():
    return app.send_static_file("index.html")


@app.route('/randomForest', methods=['POST'])
def random_forest():
    if request.method == 'POST':
        X_test_input = []
        age = request.form['age']
        X_test_input.append(age)
        salary = request.form['salary']
        X_test_input.append(salary)
        data = dict()
        data['prediction'] = randforest(X_test_input)
        return jsonify(data)


@app.route('/naiveBayes', methods=['POST'])
def bayes():
    if request.method == 'POST':
        X_test_input = []
        age = request.form['age']
        X_test_input.append(age)
        salary = request.form['salary']
        X_test_input.append(salary)
        data = dict()
        data['prediction'] = naivebayes(X_test_input)
        return jsonify(data)


@app.route('/knn', methods=['POST'])
def knn():
    if request.method == 'POST':
        X_test_input = []
        age = request.form['age']
        X_test_input.append(age)
        salary = request.form['salary']
        X_test_input.append(salary)
        data = dict()
        data['prediction'] = k_nearest_neighbours(X_test_input)
        return jsonify(data)


@app.route('/decisionTree', methods=['POST'])
def decisionTree():
    if request.method == 'POST':
        X_test_input = []
        age = request.form['age']
        X_test_input.append(age)
        salary = request.form['salary']
        X_test_input.append(salary)
        data = dict()
        data['prediction'] = decision_tree(X_test_input)
        return jsonify(data)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
