import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

import 07_vizuelizacija_matrice_konfuzije

def main():
    #Kreiranje modela algoritmom DecisionTreeClassifier na nebalansiranom skupu
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    y_test_predicted = clf.predict(X_test)
    #cuvanje modela
    pickle.dump(clf, open('drvo_sa_nebalansiranim podacima.sav', 'wb'))

    #ispis matrice konfuzije
    cm = confusion_matrix(y_test, y_test_predicted)
    print(cm)

    #iscrtavanje matrice konfuzije u normalizovanom i nenormalizovanom obliku
    plot_confusion_matrix(cm, normalize    = False,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title        = "Confusion Matrix without normalization - Decission tree (unbalanced dataset)")

    plot_confusion_matrix(cm, normalize    = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title        = "Confusion Matrix with normalization - Decision tree (unbalanced dataset)")

    #Ispis preciznosti modela kreiranom na nebalansiranom skupi
    print("Preciznost na nebalansiranom skupu: " , clf.score(X_test, y_test))

    #Kreiranje modela algoritmom Drveta odlucivanja na balansiranom skupu podataka
    clf_balanced = DecisionTreeClassifier().fit(X_balanced_df, y_balanced_df)
    y_test_predicted_balanced = clf_balanced.predict(X_test)

    #Cuvanje modela
    pickle.dump(clf_balanced, open('drvo_sa_balansiranim podacima.sav', 'wb'))

    #Ispis matrice konfuzije
    cm_balanced = confusion_matrix(y_test, y_test_predicted_balanced)
    print(cm_balanced)

    #Iscrtavanje matrice konfuzije u normlizovanom i nenormalizovanom obliku
    plot_confusion_matrix(cm_balanced, normalize    = False,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title        = "Confusion Matrix without normalization - Decision tree (balanced dataset)")

    plot_confusion_matrix(cm_balanced, normalize    = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title        = "Confusion Matrix with normalization - Decision tree (balanced dataset)")

    #Ispis preciznosti modela kreiranom na balansiranom skupu
    print("Preciznost na balansiranom skupu:",clf_balanced.score(X_test, y_test))


if __name__ == "__main__":
    main()
