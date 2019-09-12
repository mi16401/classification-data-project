import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns

def main():
    #Ucitavnaje podataka
    df = pd.read_csv('data.csv')
    print(df.shape)

    #X-karakterisike; y-ciljni atribut
    X = df.loc[:, df.columns != 'class']
    y = df.loc[:,['class']]

    #Podela na trening i test skupove:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    print("Velicina trening skupa: ")
    print(X_train.shape, y_train.shape)

    print("Velicina test skupa:")
    print(X_test.shape,y_test.shape)

    #Kolicina instanci svake klase u trening skupu
    print(y_train.groupby('class').size())
    #class
    #cetvrta    4838
    #druga      1199
    #prva       1177
    #treca      4861

    #BALANSIRANJE SKUPA ZA TRENING
    smote = SMOTE('all')
    X_balanced, y_balanced = smote.fit_sample(X_train,y_train)

    #Posto se dobijaju nizovi umesto DataFrame objekta moramo da napravimo DataFrame-ove
    y_balanced_df = pd.DataFrame(y_balanced,columns=['class'])
    print(y_balanced_df.shape) #Nova dimenzija ciljnog atributa: (19444, 1)

    X_balanced_df = pd.DataFrame(X_balanced)
    print(X_balanced_df.shape) #Nova dimenzija ostalih atributa: (19444, 10752)

    #Kolicina instanci u svakoj klasi nakon balansiranja:
    print(y_balanced_df.groupby('class').size())
    #class
    #cetvrta    4861
    #druga      4861
    #prva       4861
    #treca      4861

    #Kreiranje histograma sa balansiranim podacima
    sns.countplot(y_balanced_df['class'],label="Count")
    plt.savefig('klase_balansirane_hist.png', bbox_inches='tight')


if __name__ == "__main__":
    main()
