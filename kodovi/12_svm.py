import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pickle

import 07_vizuelizacija_matrice_konfuzije

def main():

    #Kreiranje modela klasifikacije
    svm = SVC(kernel = 'linear')
    svm.fit(X_balanced_df, y_balanced_df)

    #Cuvanje modela
    pickle.dump(svm, open('SVM_linear.sav', 'wb')

    y_predicted_svm = svm.predict(X_test)
    cm_svm = confusion_matrix(y_test, y_predicted_svm)
    print(cm_svm)
    #Matrica konfuzije:
    #[[1637    8    0    0]
    #[  16  351    1    0]
    #[   4    0  369   17]
    #[  11    0    7 1604]]

    print("Preciznost modela dobijenog SVM algoritmom sa linearnim kernelom: ",svm.score(X_test, y_test))
    #Preciznost modela dobijenog SVM algoritmom sa linearnim kernelom: 0.9840993788819876

    #Vizuelni prikaz normalizovane matrice konfuzije
    plot_confusion_matrix(cm_svm, normalize = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title = "Confusion Matrix - SVM (kernel = 'linear')")

if __name__ == "__main__":
    main()
