import pandas as pd

def main():
    #UCITAVANJE, TRANSPONOVANJE I DODAVANJE ATRIBUTA KLASE U SVAKOJ OD CETIRI DATOTEKE
    df_003 = pd.read_csv('003.csv',index_col = 0)
    df_trans_003 = df_003.transpose()
    df_trans_003['class'] = "prva"

    df_004 = pd.read_csv('004.csv',index_col = 0)
    df_trans_004 = df_004.transpose()
    df_trans_004['class'] = "druga"

    df_005 = pd.read_csv('005.csv',index_col = 0)
    df_trans_005 = df_005.transpose()
    df_trans_005['class'] = "treca"

    df_006 = pd.read_csv('006.csv',index_col = 0)
    df_trans_006 = df_006.transpose()
    df_trans_006['class'] = "cetvrta"

    #SPAJANJE SVIH MATRICA U JEDNU
    data = pd.concat([df_trans_003,df_trans_004,df_trans_005,df_trans_006])
    print(data.shape) #dimenizija ukupne matrice: (16100,31222)

    #MATRICU SA SPOJENIM PODACIMA CUVAMO U NOVOJ DATOTECI RADI KASNIJEG PONOVNOG KORISCENJA
    data.to_csv(r'bigdata.csv')

if __name__ == "__main__":
    main()
