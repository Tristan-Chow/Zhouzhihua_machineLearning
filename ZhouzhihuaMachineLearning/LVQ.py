import numpy as np

x = np.array([[.697, .46], [.774, .376], [.634, .264], [.608, .318],
              [.556, .215], [.403, .237], [.481, .149], [.437, .211],
              [.666, .091], [.243, .267], [.245, .057], [.343, .099],
              [.639, .161], [.657, .198], [.36, .37], [.593, .042],
              [.719, .103]])

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def Learning_vector_Quantization(q, x, y, lamda, maxiter):
    data_size = x.shape[0]
    data_dimension = x.shape[1]
    principal_vec = []
    principal_label = []
    while len(principal_vec) < q:
        index = np.random.randint(0, data_size - 1, 1)
        if len(principal_vec) == 0:
            principal_vec.append(x[index][-1].tolist())
            principal_label.append(y[index][-1])
        if x[index][-1].tolist() not in principal_vec:
            principal_vec.append(x[index][-1].tolist())
            principal_label.append(y[index][-1])
    count = 0
    while count < maxiter:
        i = np.random.randint(0, data_size-1, 1)
        nearest_vector = -1
        nearest_distance = float('inf')
        for j in range(len(principal_vec)):
            distance = np.sqrt(np.sum(np.square(x[i] - np.array(principal_vec[j]))))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_vector = j
        if y[i] == principal_label[j]:
            principal_vec[nearest_vector] += np.array(principal_vec[nearest_vector]) + lamda * (
            x[i] - np.array(principal_vec[nearest_vector]))

        else:
            principal_vec[nearest_vector] -= np.array(principal_vec[nearest_vector]) + lamda * (
            x[i] - np.array(principal_vec[nearest_vector]))
            principal_label[j] = y[i]
        principal_vec[nearest_vector] = principal_vec[nearest_vector].tolist()
        count += 1
    return principal_vec
print(Learning_vector_Quantization(3, x, y, 0.1, 100))