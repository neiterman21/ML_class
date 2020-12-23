import numpy as np


class Rule:
    def __init__(self, x, y, lab, noY=False):
        self.x = x
        self.y = y
        self.lab = lab
        self.noY = noY

    def predict(self, p):
        """
        Using the current rule, apply it to new point p.
        """
        prediction = 0
        above_line = p[1] >= self.x * p[0] + self.y
        if self.noY:
            above_line = p[1] >= self.x
        if above_line:
            prediction = 1
        else:
            prediction = -1
        prediction = prediction * self.lab
        return prediction
