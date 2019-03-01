import numpy as np

x = np.array([[.697, .46], [.774, .376], [.634, .264], [.608, .318],
              [.556, .215], [.403, .237], [.481, .149], [.437, .211],
              [.666, .091], [.243, .267], [.245, .057], [.343, .099],
              [.639, .161], [.657, .198], [.36, .37], [.593, .042],
              [.719, .103]])


def kmeans(k, data):
    data_size = data.shape[0]
    data_dimension = data.shape[1]
    center_point = []
    while len(center_point) < k:
        index = np.random.randint(0, data_size - 1, 1)
        if len(center_point) == 0:
            center_point.append(data[index][-1].tolist())
        if data[index][-1].tolist() not in center_point:
            center_point.append(data[index][-1].tolist())
    while True:
        cluster = [[] for _ in range(k)]
        for i in range(data_size):
            nearest_center = -1
            nearest_distance = float('inf')
            for j in range(len(center_point)):
                distance = np.sqrt(np.sum(np.square(data[i] - np.array(center_point[j]))))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_center = j
            cluster[nearest_center].append(data[i].tolist())
        new_centerpoint = []
        for sub_cluster in cluster:
            sub_size = len(sub_cluster)
            point_sum = np.array([0.0 for _ in range(data_dimension)])
            for i in range(sub_size):
                point_sum += np.array(sub_cluster[i])
            new_centerpoint.append((point_sum / sub_size).tolist())
        if new_centerpoint == center_point:
            break
        else:
            center_point = new_centerpoint
    return center_point


center_point = kmeans(3, x)
print(center_point)
