import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

import 07_vizuelizacija_matrice_konfuzije

def main():
    #LOGISTICKA REGRESIJA
    logreg = LogisticRegression()
    logreg.fit(X_balanced, y_balanced)
    #Cuvanje modela
    pickle.dump(logreg, open('logreg_model.sav', 'wb'))

    #Ispis matrice konfuzije:
    y_predicted = logreg.predict(X_test)
    cm_logreg = confusion_matrix(y_test, y_predicted)
    print(cm_logreg)

    #Matrica konfuzije:
    #[[1641    4    0    0]
    #[  12  355    1    0]
    #[   0    2  372   16]
    #[  10    0    8 1604]]

    #Vizuelni prikaz matrice konfuzije:
    plot_confusion_matrix(cm_logreg, normalize    = True,
                      target_names = ['cetvrta' , 'druga', 'prva', 'treca'],
                      title = "Confusion Matrix -Logistic Regression")

    #Preciznost modela:
    print("Preciznost modela dobijenog logistickom regresijom:",logreg.score(X_test, y_test))
    #Preciznost modela dobijenog logistickom regresijom: 0.986832298136646

if __name__ == "__main__":
    main()
