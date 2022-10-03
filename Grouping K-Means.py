# Import the necessary libraries
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from kneed import KneeLocator
import datetime

# Read the data provided
data = pd.read_csv('person_data.csv')

# Input from the user, what to take into consideration when forming the groups
cols = ['Experience (Years)', 'Specialty', 'Major']
data2 = data[cols]

def find_num_clusters(min_clusters, max_clusters, data):
    # Calculate the average cost for each number of clusters
    cost = []
    lowest = 1000000
    K = range(1,10)
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
        kmode.fit_predict(data)
        if kmode.cost_ < lowest:
            lowest = kmode.cost_
        cost.append(kmode.cost_)

    # Elbow curve to find optimal number of clusters
    kneedle = KneeLocator(K, cost, S=1.0, curve="convex", direction="decreasing")
    return kneedle.elbow, lowest

def find_initial_centroids(lowest, n_clusters, data):
    # Find the lowest cost for each group of initial centroids
    # and return the best one
    current = lowest * 1.5
    time_before = datetime.datetime.now()
    while (current > lowest * 1.15):
        kmode = KModes(n_clusters=n_clusters, init = "random", n_init = 5, verbose=1)
        kmode.fit_predict(data)
        if kmode.cost_ < current:
            current = kmode.cost_
            centroids = kmode._enc_cluster_centroids
        if (datetime.datetime.now() - time_before).seconds > 30:
            break
    return centroids

def make_clusters(n_clusters, centroids, data):
    # Build the model with the best number of clusters
    # and the initial centroids selected
    kmode = KModes(n_clusters=n_clusters, init = centroids, n_init = 5, verbose=1)
    clusters = kmode.fit_predict(data)
    
    return clusters

def group_people(data, groups):
    # Iterate through each person in the dataframe and assign them to
    # a group based on the cluster they are in
    person_index = 0
    
    while person_index < data.shape[0]:
        for i in range(0, len(groups)):
            if person_index >= data.shape[0]:
                break
            data.loc[person_index, 'Group'] = groups[i]
            person_index += 1
        i = 0

    return data

n_clusters, lowest_cost = find_num_clusters(1, 10, data=data2)

best_centroids = find_initial_centroids(lowest=lowest_cost, n_clusters=n_clusters, data=data2)

data['Cluster'] = make_clusters(n_clusters=n_clusters, centroids=best_centroids, data=data2)

data.to_csv('clustered_data.csv')

clustered_data = pd.read_csv('clustered_data.csv')
clustered_data.drop(columns=['Unnamed: 0'], inplace=True)

# User input: number of people per group
people_group = 5

# Find number of groups and make a list of those groups
rows = clustered_data.shape[0]
number_groups = int(rows/people_group)
groups = list(range(1, number_groups))

clustered_data = clustered_data.sort_values(by='Cluster')
clustered_data['Group'] = -1

grouped_data = group_people(data=clustered_data, groups=groups)

grouped_data.sort_values(by='Group',inplace=True)

clustered_data.drop(columns=['Cluster'], inplace=True)

grouped_data.to_csv('grouped_data.csv')