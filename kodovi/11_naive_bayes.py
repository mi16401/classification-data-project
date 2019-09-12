import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pickle

import 07_vizuelizacija_matrice_konfuzije

def main():
    #Naivni Bajesov algoritam nad balansiranim trening skupom
    gnb = GaussianNB()
    gnb.fit(X_balanced_df, y_balanced_df)

    #Cuvanje modela
    pickle.dump(gnb, open('naivni_bajes_balansirani_podaci.sav', 'wb'))

    #Kreiranje i ispis matrice konfuzije
    y_predicted_gnb = gnb.predict(X_test)
    cm_gnb = confusion_matrix(y_test, y_predicted_gnb)
    print(cm_gnb)
        #Matrice konfuzija modela dobijenog nad nebalansiranim trening skupom:
        #[[1022  536   57   30]
        #[  21  299   42    6]
        #[   0    0  389    1]
        #[   2    1  351 1268]]

    print("Preciznost modela nad balansiranim podacima:",gnb.score(X_test, y_test))
    #Preciznost modela nad balansiranim podacima: 0.7398757763975156

    #Vizuelizacija matrice konfuzije:
    plot_confusion_matrix(cm_gnb, normalize = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title = "Confusion Matrix - Naive Bayes (balanced dataset)")



    #Naivni Bajesov algoritam nad nebalansiranim trening skupom
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    pickle.dump(gnb, open('naivni_bajes_nebalansirani_podaci.sav', 'wb'))

    y_predicted_gnb = gnb.predict(X_test)
    cm_gnb = confusion_matrix(y_test, y_predicted_gnb)
    print(cm_gnb)
        #Matrice konfuzija modela dobijenog nad nebalansiranim trening skupom:
        #[[1553   11    1   80]
        # [   0  346    1   21]
        # [   0    0  373   17]
        # [   3    2    3 1614]]

    print("Preciznost modela nad nebalansiranim podacima:",gnb.score(X_test, y_test))
    #Preciznost modela nad nebalansiranim podacima: 0.9654658385093168

    #Vizuelizacija matrice konfuzije:
    plot_confusion_matrix(cm_gnb, normalize = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title = "Confusion Matrix - Naive Bayes (unbalanced dataset)")

if __name__ == "__main__":
    main()
