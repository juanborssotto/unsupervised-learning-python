import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

# [Weight, Height]
X = np.array([[55, 155],
             [56, 160],
             [70, 185],
             [74, 180],
             [67, 174],
             [60, 167]])

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r.', 'g.', '.b']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.ylabel('Height')
plt.xlabel('Weight')

# Prediction
predict_me = np.array([58, 185])
predict_me = predict_me.reshape(-1, len(predict_me))
plt.plot(predict_me[0][0], predict_me[0][1], colors[clf.predict(predict_me)[0]], markersize=10,)
plt.annotate('Prediction', (predict_me[0][0], predict_me[0][1]))
plt.show()