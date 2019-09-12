import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pickle

import 07_vizuelizacija_matrice_konfuzije

def main():

    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',algorithm = 'brute')
    knn.fit(X_balanced, y_balanced)

    pickle.dump(knn, open('knn_k_5_minkowski_brute.sav', 'wb'))

    y_predicted_knn = knn.predict(X_test)
    cm_knn = confusion_matrix(y_test, y_predicted_knn)
    print(cm_knn)
    #Matrica konfuzije
    #[[1251  326   25   43]
    #[  31  324   13    0]
    #[   8   50  319   13]
    #[ 330   59  249  984]]

    print("Preciznost modela dobijenog metodom k najblizih suseda:",knn.score(X_test, y_test))
    #"Preciznost modela dobijenog metodom k najblizih suseda:  0.7150310559006211

    plot_confusion_matrix(cm_knn, normalize    = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title  = "Confusion Matrix - kNN (k = 4; metric = 'minkowski', algorithm = 'brute')")


if __name__ == "__main__":
    main()
