import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    #ucitavanje podataka koji su dobijeni nakon spajanja i eliminacije nula-kolona
    df = pd.read_csv("data.csv")

    #prikaz broja instanci u svakoj klasi na histogramu
    sns.countplot(df['class'],label="Count")

    #cuvanje histograma za kasniju upotrebu u izvestaju
    plt.savefig('klase_hist.png', bbox_inches='tight')

if __name__ == "__main__":
    main()
