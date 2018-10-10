import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict
import math

df_nn = pd.read_csv('games.csv')
df = pd.read_csv('games.csv')
df.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'Rating'], 1, inplace=True)

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

num_clusters=6
clf = KMeans(n_clusters=num_clusters)
clf.fit(df)
labels = clf.labels_

for j in range(num_clusters):
    publishers_count = defaultdict(int)
    avg_global_sales = 0.0
    avg_critic_score = 0.0
    c = 0
    for i in range(len(df)):
        if labels[i] == j:
            publishers_count[df_nn['Publisher'][i]]+= 1
            c+= 1
            avg_global_sales+= df_nn['Global_Sales'][i]
            avg_critic_score+= df_nn['Critic_Score'][i] if not math.isnan(df_nn['Critic_Score'][i]) else 0.0
    print('')
    print('Cluster '+ str(j))
    print('Avg global sales '+ str(avg_global_sales / c) + ' millions copies')
    print('Avg Critic Score ' + str(avg_critic_score / c))
    print('Top publisher ' + str(max(publishers_count, key=publishers_count.get)))

plat_to_pred = '3DS'
year_of_rel_to_pred = 2011
genre_to_pred = 'Racing'
publ_to_pred = 'Nintendo'
usr_score_to_pred = 7.0
usr_count_to_pred = 924
for i in range(len(df)):
    if(df_nn['Platform'][i] == plat_to_pred):
        pred_plat = df['Platform'][i]
    if(df_nn['Genre'][i] == genre_to_pred):
        pred_genre = df['Genre'][i]
    if(df_nn['Publisher'][i] == publ_to_pred):
        pred_publ = df['Publisher'][i]

predict_me = [[pred_plat, year_of_rel_to_pred, pred_genre, pred_publ, usr_score_to_pred, usr_count_to_pred]]
prediction = clf.predict(predict_me)
print('')
print('Prediction with values:')
print('Platform: {0}'.format(plat_to_pred))
print('Year of release: {0}'.format(year_of_rel_to_pred))
print('Genre: {0}'.format(genre_to_pred))
print('Publisher: {0}'.format(publ_to_pred))
print('Avg user score: {0}'.format(usr_score_to_pred))
print('User count: {0}'.format(usr_count_to_pred))
print('')
print('Belongs to cluster: {0}'.format(prediction[0]))