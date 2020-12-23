# using python3.8
# need to install numpy and pandas
# Iris Data H_k:
"""
iris dataset num of rules = 1 Empirical error: 10.84 True error: 16.14
iris dataset num of rules = 2 Empirical error: 10.86 True error: 16.12
iris dataset num of rules = 3 Empirical error: 9.55 True error: 17.21
iris dataset num of rules = 4 Empirical error: 10.11 True error: 16.03
iris dataset num of rules = 5 Empirical error: 8.41 True error: 17.21
iris dataset num of rules = 6 Empirical error: 8.75 True error: 16.67
iris dataset num of rules = 7 Empirical error: 7.09 True error: 17.86
iris dataset num of rules = 8 Empirical error: 7.6 True error: 16.83
"""
# HC Data H_k:
"""
HC_Body dataset num of rules = 1 Empirical error: 23.64 True error: 30.99
HC_Body dataset num of rules = 2 Empirical error: 24.95 True error: 30.89
HC_Body dataset num of rules = 3 Empirical error: 20.7 True error: 30.2
HC_Body dataset num of rules = 4 Empirical error: 21.64 True error: 30.67
HC_Body dataset num of rules = 5 Empirical error: 18.5 True error: 29.89
HC_Body dataset num of rules = 6 Empirical error: 19.06 True error: 30.84
HC_Body dataset num of rules = 7 Empirical error: 16.92 True error: 30.35
HC_Body dataset num of rules = 8 Empirical error: 17.56 True error: 30.46
"""
'''
Overfitting is clearly present in both results, during training the true error was going down,
but at a certain point, we added too many rules and the model became too complex.
The true error is higher than the empirical,
which means that the model didn't generalize to the true data, it learned the
training data really well but didn't classify new instances as it should.
This is because points that are not caught by the best rules increase
their weight exponentialy.
'''

##############################
from Adaboost import AdaBoost
import readData
from lineClassifier import Rule
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


def get_rules(data):
    """
    Each rule is built by iterating over the dataset and finding the best seperator.
    """
    rules = list()
    for fp in data.values:
        for sp in data.values:
            if not np.array_equal(fp, sp):
                avg1 = np.average([fp[0], sp[0]])
                avg2 = np.average([fp[1], sp[1]])
                sameX = fp[0] == sp[0]
                sameY = fp[1] == sp[1]
                mid_point = [avg1, avg2]
                lab = [1, -1]
                for l in lab:
                    if sameX:
                        rules.append(Rule(0, mid_point[1], l))
                    elif sameY:
                        rules.append(Rule(mid_point[0], 0, l, True))
                    else:
                        y_diff = fp[1] - sp[1]
                        x_diff = fp[0] - sp[0]
                        x = -(1 / (y_diff / x_diff))
                        y = mid_point[1] - x * mid_point[0]
                        rules.append(Rule(x, y, l))
    return rules


def calc_error(rules, data):
    err = 0
    for i in range(1, len(data.index)):
        w_vote = 0
        for rule in rules:
            w_vote += rule[1] * rule[0].predict([data["c1"][i], data["c2"][i]])
        if np.sign(w_vote) != data["label"][i]:
            err += 1
    return err


def run_single_experiment(data, rules):
    """
    Perform a single iteration of 8 runs.
    """
    train_data, test_data = np.vsplit(data.sample(frac=1).reset_index(drop=True), 2)
    test_data = test_data.reset_index(drop=True)
    classifier = AdaBoost(rules, 8)
    best_rules = classifier.fit(train_data)
    emp_error = []
    true_error = []
    for k in range(1, 1 + len(best_rules)):
        emp_error.append(calc_error(best_rules[:k], train_data))
        true_error.append(calc_error(best_rules[:k], test_data))
    return np.asarray(emp_error), np.asarray(true_error)


def run_experiment(data, iterations=100):
    """
    Run AdaBoost on data where every iteration 8 runs are performed.
    iterations default value is 100.
    Note that this function runs using multiple cores, adjust below according to your specs
    """
    emp_error = np.zeros(8)
    true_error = np.zeros(8)
    rules = get_rules(data)
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores)(
        delayed(run_single_experiment)(data, rules) for i in range(iterations)
    )
    for res in processed_list:
        emp_error = np.add(emp_error, res[0])
        true_error = np.add(true_error, res[1])
    emp_error /= iterations
    true_error /= iterations
    return emp_error, true_error


def main():
    iris = readData.iris_data("iris.data")
    emp_err, true_err = run_experiment(iris, 2)
    i = 1
    for emp, tru in zip(emp_err, true_err):
        print(
            "iris dataset num of rules =",
            i,
            "Empirical error:",
            emp,
            "True error:",
            tru,
        )
        i += 1
    hbt = readData.hbt_data("HC_Body_Temperature.txt")
    emp_err, true_err = run_experiment(hbt, 2)
    i = 1
    for emp, tru in zip(emp_err, true_err):
        print(
            "HC_Body dataset num of rules =",
            i,
            "Empirical error:",
            emp,
            "True error:",
            tru,
        )
        i += 1


if __name__ == "__main__":
    main()
