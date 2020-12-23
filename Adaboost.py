import numpy as np
import sys


def sign(x):
    return abs(x) / x if x != 0 else 1


class AdaBoost:
    def __init__(self, rules, iterations):
        self.rules = rules
        self.iterations = iterations

    def fit(self, data):
        """
        Compute AdaBoost on the data, the algorithm is implemented as was shown in
        Lee-Ad's Gottlieb Machine Learning course.
        """
        N = len(data.index)
        w = np.array([1 / N for i in range(N)])
        best_rules = []
        for i in range(self.iterations):
            best_error = sys.maxsize
            best_rule = self.rules[0]
            for rule in self.rules:
                weer = 0
                for i in range(1, len(data.index)):
                    if rule.predict([data["c1"][i], data["c2"][i]]) != data["label"][i]:
                        weer += w[i]
                if best_error > weer:
                    best_error = weer
                    best_rule = rule
            # fix numerical issues
            best_error = 0.999 if best_error >= 1 else best_error
            best_error = 0.001 if best_error <= 0 else best_error
            rule_w = 0.5 * np.log((1 - best_error) / best_error)
            best_rules.append((best_rule, rule_w))
            for i in range(1, w.__len__()):
                w[i] *= np.e ** (
                    -rule_w
                    * best_rule.predict([data["c1"][i], data["c2"][i]])
                    * data["label"][i]
                )
                w /= np.sum(w)
        return best_rules
