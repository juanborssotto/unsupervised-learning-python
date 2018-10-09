import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict

df_nn = pd.read_csv('games-2.csv')
df = pd.read_csv('games-2.csv')
df.drop(['Name'], 1, inplace=True)

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist() 
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)

# X = np.array(df.drop(['Year_of_Release'], 1).astype(float))
X = preprocessing.scale(df)
# y = np.array(df['survived'])

clf = KMeans(n_clusters=5)
clf.fit(X)
labels = clf.labels_

colors = ['r.', 'g.', 'b.', 'y.', 'm.']

# for i in range(len(df)):
#     plt.plot(df['Global_Sales'][i], df['Year_of_Release'][i], colors[labels[i]], markersize=5)
# plt.ylabel('Year of release')
# plt.xlabel('Genre')
# plt.show()

for j in range(len(colors)):
    publishers_count = defaultdict(int)
    avg_global_sales = 0.0
    c = 0
    for i in range(len(df)):
        if labels[i] == j:
            publishers_count[df_nn['Publisher'][i]]+= 1
            c+= 1
            avg_global_sales+= df_nn['Global_Sales'][i]
    print('Cluster '+ str(j))
    print('Avg global sales '+ str(avg_global_sales / c))
    print('Top publisher ' + str(max(publishers_count, key=publishers_count.get)))

# for i in range(len(df)):
#     plt.plot( df['Year_of_Release'][i],df['Publisher'][i], colors[labels[i]], markersize=5)
# plt.ylabel('Sex')
# plt.xlabel('Age')
# plt.show()

predict_me = np.array(X[0].astype(float))
print(predict_me)

# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = clf.predict(predict_me)
#     if prediction[0] == y[i]:
#         correct+=1
# print(correct)
# print(len(X))
# print(float(correct)/float(len(X)))