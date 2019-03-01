import numpy as np
import math

x = np.array([[.697, .46], [.774, .376], [.634, .264], [.608, .318],
              [.556, .215], [.403, .237], [.481, .149], [.437, .211],
              [.666, .091], [.243, .267], [.245, .057], [.343, .099],
              [.639, .161], [.657, .198], [.36, .37], [.593, .042],
              [.719, .103]])

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])


class AdaBoost:
    def __init__(self, data, label, classifier, maxiter):
        self.data = data
        self.y = label
        self.classifier = classifier
        self.maxiter = maxiter
        self.classifier_weight = []
        self.classifier_array = []

    def fit(self):
        datasize = self.data.shape[0]
        data_dimension = self.data.shape[1]
        initial_weight = [1 / datasize for _ in range(datasize)]
        for i in range(self.maxiter):
            clf = self.classifier.fit(self.data, self.y)
            self.classifier_array.append(clf)
            predicted_value = clf.predict(self.data)
            error_rate = self.calculate_error(self.y, predicted_value, initial_weight)
            alpha = math.log2((1 - error_rate) / error_rate)
            self.classifier_weight.append(alpha)
            weight_total = 0
            for k in range(datasize):
                weight_total += initial_weight[k] * math.exp(-alpha * self.y[k] * predicted_value[k])
            for j in range(datasize):
                initial_weight[j] = initial_weight[j] * math.exp(-alpha * self.y[j] * predicted_value[j])

    def calculate_error(self, y, predicted_value, initial_weight):
        error_rate = 0
        for i in range(y.shape[0]):
            if y[i] != predicted_value[i]:
                error_rate += initial_weight[i]
        return error_rate

    def coefficient(self):
        return self.classifier_weight

    def predict(self, weight, classifier_array):
        result_array = []
        for classifier in classifier_array:
            result = classifier.predict(self.data)
            result_array.append(result.tolist())
        #generate final_result