import numpy as np
import math

x = np.array([[.697, .46], [.774, .376], [.634, .264], [.608, .318],
              [.556, .215], [.403, .237], [.481, .149], [.437, .211],
              [.666, .091], [.243, .267], [.245, .057], [.343, .099],
              [.639, .161], [.657, .198], [.36, .37], [.593, .042],
              [.719, .103]])

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def calculate_poster(j, data_item, alpha_set, mean_vec, variance_vec):
    # wait to be filled
    upper_probablity = (alpha_set[j] * np.exp(
        -(1.0 / 2) * np.dot(np.dot(data_item - np.array(mean_vec[j]), np.linalg.inv(variance_vec[j])),
                            (data_item - np.array(mean_vec[j])).T))) / (
                           math.power(2 * math.pi, variance_vec[j].shape[0] / 2) * math.sqrt(
                               np.linalg.det(variance_vec[j])))
    lower_probablity = 0
    for i in range(len(alpha_set)):
        lower_probablity += (alpha_set[i] * np.exp(
            -(1.0 / 2) * np.dot(np.dot(data_item - np.array(mean_vec[i]), np.linalg.inv(variance_vec[i])),
                                (data_item - np.array(mean_vec[i])).T))) / (
                                math.power(2 * math.pi, variance_vec[i].shape[0] / 2) * math.sqrt(
                                    np.linalg.det(variance_vec[i])))
    return upper_probablity / lower_probablity


def EM(k, x, y):
    data_size = x.shape[0]
    data_dimension = x.shape[1]
    alpha_set = [1 / k for i in range(k)]
    mean_vec = []
    variance_matirx = np.random.rand(data_dimension, data_dimension)
    variance_vec = [variance_matirx for _ in range(k)]
    while len(mean_vec) < k:
        index = np.random.randint(0, data_size - 1, 1)
        if len(mean_vec) == 0:
            mean_vec.append(x[index][-1].tolist())
        if x[index][-1].tolist() not in mean_vec:
            mean_vec.append(x[index][-1].tolist())
    posterior_pro = np.zeros([data_size, k])
    for i in range(data_size):
        for j in range(k):
            posterior_pro[i][j] = calculate_poster(j, x[i], alpha_set, mean_vec, variance_vec)
