import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    #Kreiranje modela algoritmom DecisionTreeClassifier na nebalansiranom skupu
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    y_test_predicted = clf.predict(X_test)
    #cuvanje modela
    pickle.dump(clf, open('drvo_sa_nebalansiranim podacima.sav', 'wb'))

    #ispis matrice konfuzije
    cm = confusion_matrix(y_test, y_test_predicted)
    print(cm)

    #iscrtavanje matrice konfuzije
    ax= plt.subplot()
    plt.figure(figsize = (20,10))
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='YlGnBu');

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['prva', 'druga','treca','cetvrta']);
    ax.yaxis.set_ticklabels(['prva', 'druga','treca','cetvrta']);



if __name__ == "__main__":
    main()
