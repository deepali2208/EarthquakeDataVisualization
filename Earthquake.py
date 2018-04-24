from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import copy
from scipy.spatial import distance
from flask import Flask,request,redirect,render_template,make_response
from scipy.spatial.distance import cdist

app=Flask(__name__)
@app.route('/')
def main():
    return render_template('cluster.html')


@app.route('/cluster', methods=['GET','POST'])
def cluster():


    #Reading data from file
    p = pd.read_csv("testdata.csv")
    headers= list(p.columns.values)
    #q is reading number of rows from 0 to 500
    q = p[0:500]

    #Implementation of KMeans Algorithm
    features = ['depth','latitude']
    p_features = q[features]
    number_of_clusters = 4
    kmeans =KMeans(n_clusters=number_of_clusters)
    kmeans.fit(p_features)
    #Calculating centroid
    centroid = kmeans.cluster_centers_
    #Calculating label
    label = kmeans.labels_

@app.route( '/scatterplot', methods=['GET', 'POST'] )
def Scatterplot():

    # Reading data from file
    p = pd.read_csv( "testdata.csv" )
    headers = list( p.columns.values )
    # q is reading number of rows from 0 to 500
    q = p[0:500]

    # Implementation of KMeans Algorithm
    features = ['depth', 'latitude']
    p_features = q[features]
    number_of_clusters = 4
    kmeans = KMeans( n_clusters=number_of_clusters )
    kmeans.fit( p_features )
    # Calculating centroid
    centroid = kmeans.cluster_centers_
    # Calculating label
    label = kmeans.labels_

    #Scatterplot for multiple series
    plt.scatter(p_features.depth,p_features.latitude,color=['blue','green'],marker='*')
    plt.scatter(centroid[:,0],centroid[:,1],color=['red','pink'],marker='X')
    r = plt.show()
    print r


@app.route( '/count', methods=['GET', 'POST'] )
def count():
    # Reading data from file
    p = pd.read_csv( "testdata.csv" )
    headers = list( p.columns.values )
    # q is reading number of rows from 0 to 500
    q = p[0:500]

    # Implementation of KMeans Algorithm
    features = ['depth', 'latitude']
    p_features = q[features]
    number_of_clusters = 4
    kmeans = KMeans( n_clusters=number_of_clusters )
    kmeans.fit( p_features )
    # Calculating centroid
    centroid = kmeans.cluster_centers_
    # Calculating label
    label = kmeans.labels_

    #Calculating number of points in a cluster
    x = Counter(label)
    return str(x)

@app.route( '/maximumdistance', methods=['GET', 'POST'] )
def maximumdistance():
    # Reading data from file
    p = pd.read_csv( "testdata.csv" )
    headers = list( p.columns.values )
    # q is reading number of rows from 0 to 500
    q = p[0:500]

    # Implementation of KMeans Algorithm
    features = ['depth', 'latitude']
    p_features = q[features]
    number_of_clusters = 4
    kmeans = KMeans( n_clusters=number_of_clusters )
    kmeans.fit( p_features )
    # Calculating centroid
    centroid = kmeans.cluster_centers_
    # Calculating label
    label = kmeans.labels_

    #Calculating distance between 2 clusters
    result = []
    for i in range( number_of_clusters ):
        for j in range( i + 1, number_of_clusters ):
            dist = distance.euclidean( centroid[i], centroid[j] )
            print(dist, centroid[i], centroid[j])
        result.append( dist )
        a = max(result)
        print a

@app.route( '/barchart', methods=['GET', 'POST'] )
def barchart():
    p = pd.read_csv( "testdata.csv" )
    q = p[0:200]
    features = ['latitude', 'longitude']
    p_features = q[features]
    second_list = []
    third_list = []
    fourth_list = []
    second_list = copy.deepcopy( p_features['latitude'].tolist() )
    third_list = copy.deepcopy( p_features['longitude'].tolist() )

    fourth_list.append( second_list )
    fourth_list.append( third_list )
    label = []
    number_of_clusters = 5
    KM = KMeans( n_clusters=5 ).fit( p_features )
    lbl = KM.labels_
    centroids = KM.cluster_centers_
    label = lbl.tolist()

    count = 0
    c = []
    datapointx = []
    datapointy = []
    for i in range( 0, number_of_clusters ):
        b = []
        a = []
        x = []

        for j in range( 0, len( lbl ) ):
            if lbl[j] == i:
                variable1 = second_list[j]
                variable2 = third_list[j]
                dt = (variable1, variable2)
                dt1 = (variable1)
                dt2 = (variable2)
                b.append( dt )
                a.append( dt1 )
                x.append( dt2 )
        c.append( b )

        datapointx.append( a )

        datapointy.append( x )

    print(datapointy[0])
    print(datapointx[0])
    print(datapointy[1])
    print(datapointx[1])
    print(c[0])
    print(c[1])
    print(c[2])
    print(c[3])
    print(c[4])
    num = []
    for i in range( 0, 5 ):
        num.append( label.count( i ) )

    # can plot specifically, after just showing the defaults:
    plt.scatter( datapointx[0], datapointy[0], linewidth=5, color='red' )
    plt.scatter( datapointx[1], datapointy[1], linewidth=5, color='blue' )
    plt.scatter( datapointx[2], datapointy[2], linewidth=5, color='green' )
    plt.scatter( datapointx[3], datapointy[3], linewidth=5, color='yellow' )
    plt.scatter( datapointx[4], datapointy[4], linewidth=5, color='violet' )

    plt.title( ' Info ' )
    plt.ylabel( 'Y axis' )
    plt.xlabel( 'X axis' )

    plt.show()
    return render_template('barchart.html',data = num)


@app.route( '/piechart', methods=['GET', 'POST'] )
def piechart():
    return render_template('piechart.html')

    """
     for i in range( 0, clusterValue ):
        maxColumnList.append( max( datapointx[i] ) )
        maxYRange.append( str( min( datapointy[i] ) ) )
        maxYRange.append( str( max( datapointy[i] ) ) )
    return render_template( 'piechart.html', data=maxColumnList, Yrange=maxYRange ) 
    """

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8082,debug=True)