# Alexandre Moro Rialp (1527046)


import numpy as np
import sklearn as sk
import matplotlib as mp
import scipy as scp
import pandas as pd
# %matplotlib notebook
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import model_selection
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Repo creat
# ----------------------------------------------------------------------------------------------------------------- #
# Primera part C EDA
# ----------------------------------------------------------------------------------------------------------------- #
def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")
"""
dataset = load_dataset("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/Melbourne_housing_FULL.csv")
# dataset = load_dataset("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/MELBOURNE_HOUSE_PRICES_LESS.csv")

#######Datasets Dimensionalitat###########
data = dataset.values

x_housing= data[:, :-1]
y_housing = data[:, -1]
print("Dimensionalitat de la BBDD_housing:", dataset.shape)
print("Dimensionalitat de les entrades x_housing", x_housing.shape)
print("Dimensionalitat de l'atribut Y_housing", y_housing.shape)

#######Datasets atributs types###########

#Posa columna Price última
atributs = dataset.columns.tolist()
aux = atributs[0:4]
aux += atributs[5:-1] #al posar -1 ja eliminem la última de property count
aux.append(atributs[4])
dataset = dataset[aux]
print(dataset.dtypes)
# print(dataset.info())
print(dataset.head())
print(dataset.describe())

#######Selecció d'Atributs###########
#Hi ha atributs redundants que aporten la mateixa informació, la localització:
# ('Suburb','Address','Distance','Postcode','CouncilArea','Latitude','Longitude', 'Regionname')
# El millor utilitzar Latitud i Longitud => Problema: té 8k Nan sobre 35k mostres => Solució: Mitjana lat i long segons suburb (hi ha 175 dif[el que més especific])

# Substitueix els NaN de Latitud i Longitud per l'average del seu suburb
def find_location(dataset):
    # dataset['Suburb']=
    suburbs = {}
    for index,s in enumerate(dataset['Suburb']):
        if s not in suburbs:
            suburbs[s] = [[],[]]
        else:
            if pd.isna(dataset['Lattitude'][index]) == False:
                suburbs[s][0].append(dataset['Lattitude'][index])
                suburbs[s][1].append(dataset['Longtitude'][index])
    #Esborrem els suburbs sense latt ni long pk no podem inventar-nos-els
    suburbs_cp = suburbs.copy(); e=[];
    aux = dataset.copy()
    for s in suburbs_cp:
        if len(suburbs[s][0]) == 0:
                suburbs.pop(s)
                e.append(s)
    counter = 0
    for i,s in enumerate(aux['Suburb']):
        if s in e:
            a = dataset.iloc[0:i-counter,]
            b = dataset.iloc[i+1-counter:,]
            dataset = a.append(b)
            counter = counter+1

    dataset = dataset.reset_index()
    for a in dataset['Suburb']:
        if a in e:
            print("No esborrats tots els Nan")

    #trobo la mitjana de cada subrurb
    keys = suburbs.keys()
    for k in keys:
        suburbs[k][0] = sum(suburbs[k][0]) / len(suburbs[k][0])
        suburbs[k][1] = sum(suburbs[k][1]) / len(suburbs[k][1])

    #substitueixo els Nan per la mitjana del suburb
    for index,s in enumerate(dataset['Suburb']):
        if pd.isna(dataset['Lattitude'][index]):
                dataset['Lattitude'][index] = suburbs[s][0]
                dataset['Longtitude'][index] = suburbs[s][1]
    return dataset

dataset = find_location(dataset)

# Substitueix els NaN de YearBuilt i Landsize per l'average del seu suburb
def find_yearBuilt_and_landsize(dataset):

    # Calculem la mitjana de YearBuilt i landsize segons el suburb
    suburbs = {}
    for index, s in enumerate(dataset['Suburb']):
        if s not in suburbs:
            suburbs[s] = [[], []]
        else:
            if pd.isna(dataset['YearBuilt'][index]) == False and dataset['YearBuilt'][index] != 0:
                suburbs[s][0].append(dataset['YearBuilt'][index])
            if pd.isna(dataset['Landsize'][index]) == False and dataset['Landsize'][index] != 0:
                suburbs[s][1].append(dataset['Landsize'][index])

    # Esborrem els suburbs sense latt ni long pk no podem inventar-nos-els
    suburbs_cp = suburbs.copy();
    e = [];
    aux = dataset.copy()
    for s in suburbs_cp:
        if len(suburbs[s][0]) == 0 or len(suburbs[s][1]) == 0:
            suburbs.pop(s)
            e.append(s)

    counter = 0
    for i, s in enumerate(aux['Suburb']):
        if s in e:
            a = dataset.iloc[0:i - counter, ]
            b = dataset.iloc[i + 1 - counter:, ]
            dataset = a.append(b)
            counter = counter + 1

    dataset = dataset.reset_index()

    # trobo la mitjana de cada subrurb
    keys = suburbs.keys()
    for k in keys:
        suburbs[k][0] = sum(suburbs[k][0]) / len(suburbs[k][0])
        suburbs[k][1] = sum(suburbs[k][1]) / len(suburbs[k][1])

    # substitueixo els Nan per la mitjana del suburb
    for index, s in enumerate(dataset['Suburb']):
        if pd.isna(dataset['YearBuilt'][index]) or dataset['YearBuilt'][index] == 0:
            dataset['YearBuilt'][index] = suburbs[s][0]
        if pd.isna(dataset['Landsize'][index]) or dataset['Landsize'][index] == 0:
            dataset['Landsize'][index] = suburbs[s][0]

    return dataset


dataset = find_yearBuilt_and_landsize(dataset)
# print(dataset.isnull().sum())
def remove_rows_Nan_prince_bed_bath_car(dataset):
    aux = dataset.copy()
    counter = 0
    for i, s in enumerate(aux['Suburb']):
        # tmp = dataset.iloc[i:i+1,]
        if pd.isna(dataset['Price'][i]) or pd.isna(dataset['Bedroom2'][i]) or pd.isna(dataset['Bathroom'][i]) or pd.isna(dataset['Car'][i]):
        # boolean = pd.isna(tmp['Price']) | pd.isna(tmp['Bedroom2']) | pd.isna(tmp['Bathroom']) | pd.isna(tmp['Car'])
        # b1 = pd.isna(tmp['Price']) == True
        # b2 = pd.isna(tmp['Bedroom2']) == True
        # b3 = pd.isna(tmp['Bathroom']) == True
        # b4 = pd.isna(tmp['Car']) == True
        # if pd.isna(tmp['Price']) == True:
            a = dataset.iloc[0:i - counter, ]
            b = dataset.iloc[i + 1 - counter:, ]
            dataset = a.append(b)
            counter = counter + 1
    # dataset = dataset.reset_index()
    print(counter)
    return dataset

dataset = remove_rows_Nan_prince_bed_bath_car(dataset)
print(dataset.isnull().sum())
# dataset = dataset.drop(['Suburb','Address','Distance','Postcode','CouncilArea','Regionname','BuildingArea'],axis=1)
dataset = dataset.drop(['Suburb','Address','Postcode','CouncilArea','Regionname'],axis=1)
print(dataset.dtypes)
print(dataset.isnull().sum())

# 1-Aquí passar atributs Type i Method a int
# Type
print(dataset['Type'].value_counts())
def codifica_type(dataset):
    one_hot = pd.get_dummies(dataset['Type'])
    one_hot.columns = ['Type(h)', 'Type(t)', 'Type(u)']
    dataset = dataset.drop('Type', axis=1)
    dataset = dataset.join(one_hot)
    return dataset

dataset = codifica_type(dataset)

# Method
def codifica_method(dataset):
    one_hot = pd.get_dummies(dataset['Method'])
    sns.countplot(dataset['Method'])
    # El one_hot de dalt fa tantes columnes com valors diferents hi ha al camp Method. Al eliminar rows han desaparegut valors i només hi ha 5 diferents enlloc de 9
    # one_hot.columns = ['Method(S)', 'Method(SP)', 'Method(PI)', 'Method(SN)', 'Method(VB)', 'Method(PN)', 'Method(SA)','Method(W)', 'Method(SS)']
    one_hot.columns = ['Method(S)', 'Method(SP)', 'Method(PI)', 'Method(VB)', 'Method(SA)']
    dataset = dataset.drop('Method', axis=1)
    dataset = dataset.join(one_hot)
    return dataset

dataset=codifica_method(dataset)

# 2-Aquí seleccionar any
def select_year_From_date(date):
    aux = date.split("/")
    aux=aux[2]
    return int(aux)

dataset['Date'].replace({x:select_year_From_date(x) for x in dataset['Date']}, inplace=True)
print(dataset['Date'].value_counts())

# 3-Aqui convertir venedor
def convert_sellerG (dataset):
    seller = dataset['SellerG'].value_counts()
    seller.unique()
    big=seller[seller.values>1000]
    small=seller[seller.values<=50]
    medium=seller.loc[(seller.values<=1000) & (seller.values>50)]
    soldby=[]
    for each_seller in dataset['SellerG']:
        if each_seller in big.index:
            soldby.append('big')
        elif each_seller in small.index:
            soldby.append('small')
        else:
            soldby.append('medium')
    dataset['Soldby']=soldby
    one_hot = pd.get_dummies(dataset['Soldby'])
    sns.countplot(dataset['Soldby'])
    one_hot.columns= ['big','medium','small']
    dataset = dataset.drop('SellerG',axis = 1)
    dataset = dataset.drop('Soldby',axis = 1)
    dataset = dataset.join(one_hot)

    return dataset

dataset = convert_sellerG(dataset)


# Salvo les dades fins aquí per carregar després
dataset.to_csv("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/little__db_linia_243.csv")
z=3
"""

"""
dataset = load_dataset("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/little__db_linia_243.csv")

# Actualitzem els price segons IPC: 2017-1.7  2018-1.9
# Posar price al final
atributs = dataset.columns.tolist()
aux = atributs[3:14]
aux += atributs[15:]
aux.append(atributs[14])
dataset = dataset[aux]
dataset = dataset.reset_index()
# dataset = dataset.drop(['BuildingArea'])
def remove_rows_BuildingArea(dataset):
    aux = dataset.copy()
    counter = 0
    for i, s in enumerate(aux['Rooms']):
        # tmp = dataset.iloc[i:i+1,]
        if np.isnan(dataset['BuildingArea'][i]):
            a = dataset.iloc[0:i - counter, ]
            b = dataset.iloc[i + 1 - counter:, ]
            dataset = a.append(b)
            counter = counter + 1
    return dataset

dataset = remove_rows_BuildingArea(dataset)
dataset = dataset.reset_index() #ha ficat Level0 i Index

def actualize_inflation_price(dataset):
    inflation = {'2016': 1.16, '2017': 1.09, '2018': 1.0}
    for i,r in enumerate(dataset['Date']):
        dataset['Price'][i] = dataset['Price'][i]*inflation[str(r)]
    return dataset

dataset = actualize_inflation_price(dataset)


print(dataset.isnull().sum())
dataset.to_csv("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/little__db_linia_284.csv")
"""

"""
#######Datasets Tractament de NaN ###########
def check_room_same_bedrooms(dataset):
    counter_not_equal = 0
    for room,bedroom in zip(dataset['Rooms'],dataset['Bedroom2']):
        if np.isnan(bedroom) == False:
            if room == int(bedroom):
                counter_not_equal += 1
    print("Nombre de coicidents sobre 26640: {}".format(counter_not_equal))

check_room_same_bedrooms(dataset)


def check_bathroom(dataset):
    counter_more_4_1 = 0
    counter_more_4_2 = 0
    counter_less_4_1 = 0
    counter_less_4_2 = 0
    for room, bathroom in zip(dataset['Rooms'], dataset['Bathroom']):
        if np.isnan(bathroom) == False:
            if room > 4:
                if bathroom > 1:
                    counter_more_4_2 += 1
                else:
                    counter_more_4_1 += 1
            elif bathroom > 1:
                counter_less_4_1 += 1
            else:
                counter_less_4_2 += 1
    print("Nombre de more 4_1 {}".format(counter_more_4_1))
    print("Nombre de more 4_2 {}".format(counter_more_4_2))
    print("Nombre de less 4_1 {}".format(counter_less_4_1))
    print("Nombre de less 4_2 {}".format(counter_less_4_2))

check_bathroom(dataset)
# dataset= dataset.replace(np.nan,0)

dataset.to_csv("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/little__db_linia_324.csv")
"""



dataset = load_dataset("/home/alexandre/Desktop/APC/Cas Kaggle/APC_Kaggle21/data/little__db_linia_324.csv")
atributs = dataset.columns.tolist()
aux = atributs[3:]
dataset = dataset[aux]
# dataset = dataset.reset_index()
def correlacio_pearson(dataset):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))  # per mida cel·les
    plt.title("Correlació Price")
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)
    plt.show()

# correlacio_pearson(dataset)

def testeja_normalitat(dataset, algoritme):
    from scipy.stats import jarque_bera
    from scipy.stats import chisquare

    data = dataset.values
    print("Resultats normalitat")
    for i in range(dataset.shape[1]):
        x = data[:, i]
        if algoritme == "bera":
            stat, p = jarque_bera(x)
        else:
            stat, p = chisquare(x)  # aqui el Chi Square

        alpha = 0.05
        if p > alpha:
            print('Estadisticos=%.3f, p=%.3f' % (stat, p))
            print(
                'La muestra SI parece Gaussiana o Normal (no se rechaza la hipótesis nula H0)' + dataset.columns[i])
        else:
            print('La muestra NO parece Gaussiana o Normal(se rechaza la hipótesis nula H0) el atributo ' +
                  dataset.columns[i])

# testeja_normalitat(dataset, "chi")
# Tots normals excepte Landsize, BuildingArea i Price

def make_pairplot(dataset, atributs):
    plt.figure()
    sns.pairplot(dataset[atributs])
    plt.show()

atributs = dataset.columns.tolist()
# make_pairplot(dataset, atributs)


def make_pairplot_per_atribute(dataset, atributes):
    for atr in atributes:
        sns.set_theme('notebook', style='dark')
        sns.pairplot(dataset[[atr]], height=5).fig.suptitle("Correlació Gausiana: {}".format(atr), y=1)
        plt.show()

        plt.title("Correlació respecte price de {} ".format(atr))
        plt.scatter(dataset['Price'], dataset[atr])
        plt.ylabel(atr)
        plt.xlabel('Price')
        plt.show()

# make_pairplot_per_atribute(dataset,atributs)

def estandaritzar_mitjana(dataset):
    return (dataset - dataset.mean(0)) / dataset.std(0)

def make_histogrames(dataset, atributes):
    dataset_norm = estandaritzar_mitjana(dataset)
    for atr in atributes:
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Histograma de l'atribut {}".format(atr))
        ax1.hist(dataset[atr], bins=11, range=[np.min(dataset[atr]), np.max(dataset[atr])], histtype="bar", rwidth=0.8)
        ax1.set(xlabel='Attribute Value', ylabel='Count')
        ax2.hist(dataset_norm[atr], bins=11, range=[np.min(dataset_norm[atr]), np.max(dataset_norm[atr])],
                 histtype="bar", rwidth=0.8)
        ax2.set(xlabel='Normalized value', ylabel='')
        plt.show()


# make_histogrames(dataset,atributs)


# ----------------------------------------------------------------------------------------------------------------- #
################  Regressions  ##################

# atributs_correlació=['Rooms','Bedroom2','Bathroom','Car','Lattitude','Longtitude','Type(h)']
atributs_correlació=['Rooms','Bedroom2','Bathroom','Car','Distance','Type(h)']
def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)
    # Retornem el model entrenat
    return regr


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


# Dividim dades en 80% train i 20½ Cvalidation
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


def regressor_lineal(dataset):
    dataset_norm = standarize(dataset)
    data = dataset_norm.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)

    regr = regression(x_train, y_train)
    predicted = regr.predict(x_val)

    error = str(round(mse(y_val, predicted), 3))  # calculem error
    # r2 = str(round(r2_score(y_val, predicted), 3))

    errors = abs(predicted - y_val)
    mape = np.mean(100 * (errors / y_val))
    accuracy = 100 - mape
    msg = "= %.2f" % ( round(accuracy, 2))

    print('Accuracy of', msg, '%')
    print("Error: %s" % (error))
    # print("R2 score: %s" % (r2))

    # K_Fold = model_selection.KFold(n_splits=10, random_state=random.randint(0, 99), shuffle=True)
    # cv_results = model_selection.cross_val_score(LinearRegression(), x_data, y_data, cv=K_Fold)
    # message = "%s:  %f  (%f)" % ('LinealReg R2 mean:', cv_results.mean(), cv_results.std())
    # print(message)

    resultat = []; plt.figure(); res_tmp = []; i_index = [2, 3, 4, 6, 10, 15]
    for i in i_index:
        K_Fold = model_selection.KFold(n_splits=i, random_state=random.randint(0, 99), shuffle=True)
        cv_results = model_selection.cross_val_score(LinearRegression(), x_data, y_data, cv=K_Fold)
        message = "%s:  %f  (%f)" % ('LinealReg R2 mean with k-{}'.format(i), cv_results.mean(), cv_results.std())
        print(message)
        res_tmp.append(cv_results.mean())

    resultat.append(res_tmp)
    plt.plot(i_index, res_tmp, label='{}'.format('LinealReg'))
    plt.ylim(0.2, 0.5)
    plt.legend()
    plt.xlabel('Folds count')
    plt.ylabel('{}'.format('R2'))
    plt.title('Lineal Regressor {} by K-fold'.format('R2 score'))
    # plt.savefig("../figures/model_{}_kfoldB".format(type_score))
    plt.show()

regressor_lineal(dataset[atributs_correlació])


def random_forest(dataset):
    dataset_norm = standarize(dataset)
    data = dataset_norm.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)

    r_forest = RandomForestRegressor(max_depth=15)
    # R2 amb 10 k
    K_Fold = model_selection.KFold(n_splits=5, random_state=random.randint(0, 99), shuffle=True)
    cv_results = model_selection.cross_val_score(RandomForestRegressor(max_depth=15), x_data, y_data, cv=K_Fold)
    message = "%s:  %f  (%f)" % ('Random Forest R2 mean:', cv_results.mean(), cv_results.std())
    print(message)

    # Accuracy
    r_forest_fitted = r_forest.fit(x_train, y_train)
    predcited = r_forest_fitted.predict(x_val)
    errors = abs(predcited - y_val)
    mape = np.mean(100 * (errors / y_val))
    accuracy = 100 - mape
    msg = "%s= %.2f" % ('Random Forest', round(accuracy, 2))
    print('Accuracy of', msg, '%')

    #Gràfica Accuracy Train Test segons depth
    max_depths = np.linspace(1, 50, 50, endpoint=True)

    train_results = []
    test_results = []

    for i in max_depths:
        dt = RandomForestRegressor(max_depth=i)
        dt.fit(x_train, y_train)
        predicted = dt.predict(x_train)
        errors = abs(predicted - y_train)

        mape = 100 * (errors / y_train)
        accuracy = 100 - np.mean(mape)
        train_results.append(accuracy)
        predicted = dt.predict(x_val)

        errors = abs(predicted - y_val)
        mape = 100 * (errors / y_val)
        accuracy = 100 - np.mean(mape)
        test_results.append(accuracy)

    from matplotlib.legend_handler import HandlerLine2D
    print('Random Forest best accuracy:',max(test_results))
    line1, = plt.plot(max_depths, train_results, 'b', label='Train accuracy')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy score')
    plt.xlabel('Tree depth')
    plt.title('Random Forest Accuracy')
    plt.show()

random_forest(dataset[atributs_correlació])


def decision_tree(dataset):
    dataset_norm = standarize(dataset)
    data = dataset_norm.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)

    d_tree = DecisionTreeRegressor(random_state=0)
    #R2 amb 10 k
    K_Fold = model_selection.KFold(n_splits=5, random_state=random.randint(0, 99), shuffle=True)
    cv_results = model_selection.cross_val_score(DecisionTreeRegressor(random_state=10), x_data, y_data, cv=K_Fold)
    message = "%s:  %f  (%f)" % ('Decision Tree R2 mean:', cv_results.mean(), cv_results.std())
    print(message)

    #Accuracy
    d_tree_fitted = d_tree.fit(x_train,y_train)
    predcited = d_tree_fitted.predict(x_val)
    errors = abs(predcited - y_val)
    mape = np.mean(100 * (errors / y_val))
    accuracy = 100 - mape
    msg = "%s= %.2f" % ('Decision Tree',round(accuracy, 2))
    print('Accuracy of', msg, '%')

    ####Prova gràfica

    max_depths = np.linspace(1, 50, 50, endpoint=True)

    train_results = []
    test_results = []

    for i in max_depths:
        dt = DecisionTreeRegressor(max_depth=i)
        dt.fit(x_train, y_train)
        predicted = dt.predict(x_train)
        errors = abs(predicted - y_train)

        mape = 100 * (errors / y_train)
        accuracy = 100 - np.mean(mape)
        train_results.append(accuracy)
        predicted = dt.predict(x_val)

        errors = abs(predicted - y_val)
        mape = 100 * (errors / y_val)
        accuracy = 100 - np.mean(mape)
        test_results.append(accuracy)

    from matplotlib.legend_handler import HandlerLine2D
    print('Decision Tree best accuracy:', max(test_results))
    line1, = plt.plot(max_depths, train_results, 'b', label='Train accuracy')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test accuracy')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy score')
    plt.xlabel('Tree depth')
    plt.title('Decision Tree Accuracy')
    plt.show()


decision_tree(dataset[atributs_correlació])




z=3












