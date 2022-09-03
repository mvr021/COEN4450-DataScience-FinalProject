# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:43:59 2022

@author: yeya
"""

import pandas as pd
import numpy as np
from math import sqrt, pi, e
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def reducir(df, datos):
    """reduce las caracteristicas a dos columnas por medio del metodo PCA"""

    if datos == 'data_4.csv':
        df = df.replace({'yes': 1, 'no': 2, 'single': 1,
                        'married': 2, 'divorced': 3})

    label = df.iloc[:, -1]
    label = pd.DataFrame(label)
    data = df.iloc[:, :-1]
    data = pd.DataFrame(data)

    p = PCA(2)
    pca_columns = p.fit_transform(data)
    pca_columns = pd.DataFrame(pca_columns)
    df = pca_columns.join(label)
    df.columns = range(df.columns.size)

    return df


def graficar(df, clases, datos):
    """traza un grafico de dispersión"""
    scatter = plt.scatter(
        df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap='tab20b', alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(clases)
    plt.title(datos)
    plt.show()


def separar_clases(df, df_shuffled):
    """separa las clases en un diccionario"""
    
    label = df_shuffled.iloc[:, -1]

    clases_separadas = dict()
    for i in range(len(df)-1):
        clase = label[i]
        if clase not in clases_separadas:
            clases_separadas[clase] = list()
            clases_separadas[clase].append(i)
        else:
            clases_separadas[clase].append(i)

    clases = list(clases_separadas.keys())

    return label, clases_separadas, clases


def laplace(PXC, count_all_when, count_all, test):
    """aplica la técnica de suavizado Laplace que ayuda solucionar el problema de la probabilidad cero"""
    
    alpha = 1
    if PXC == 0.0:
        PXC = (count_all_when + alpha) / (count_all + alpha * test.columns[-1])

    return PXC


def calcular_media(train_column_seleccion):
    """calcula la media"""
    
    number_list = []
    for i in range(len(train_column_seleccion)):
        number = train_column_seleccion.iloc[i]
        number_list.append(number)

    media = sum(number_list) / len(number_list)

    return media


def calcular_varianza(media, train_column_seleccion):
    """calcula la varianza"""
    
    desviacion_list = []
    for i in train_column_seleccion:
        desviacion = (i - media) ** 2
        desviacion_list.append(desviacion)

    varianza = sum(desviacion_list) / len(train_column_seleccion)

    return varianza


def densidad_probabilidad(x, test, train, train_column, key, column, count_all, PXC_list, media_list, varianza_list):
    """calcula la densidad de probabilidad para caracteristicas continuas y discretas"""
    
    if type(x) == str:

        count_all_when = len(train.loc[(train[train.columns[-1]] == key) &
                                       (train_column == x)])

        PXC = count_all_when / count_all

        PXC = laplace(PXC, count_all_when, count_all, test)

        PXC_list.append(PXC)

    else:

        train_column_seleccion = train.loc[(train[train.columns[-1]] == key)]
        train_column_seleccion = train_column_seleccion.iloc[:, column+1]

        media = calcular_media(train_column_seleccion)
        media_list.append(media)

        varianza = calcular_varianza(media, train_column_seleccion)
        varianza_list.append(varianza)

        PXC = (1 / (sqrt(2 * pi * varianza))) * \
            ((e) ** ((-(x - media)**2)/(2 * varianza)))

    return PXC


def bayes(PXC_list, count_all, train):
    """calcula la probabilidad de Bayes"""
    
    PXC_result = 1.0
    for j in PXC_list:

        PXC_result = PXC_result * j

    PC = count_all / len(train)

    probabilidad = PXC_result * PC

    return probabilidad


def pertenencia(probabilidad_list, clases):
    """determina a cuál clase pertenece en base a la probabilidad"""
    
    probabilidad_max = 0
    for i in range(len(probabilidad_list)):

        number = probabilidad_list[i]
        if number > probabilidad_max:
            probabilidad_max = number

    if probabilidad_max == probabilidad_list[0]:
        prediccion = clases[0]

    elif probabilidad_max == probabilidad_list[1]:
        prediccion = clases[1]

    else:
        prediccion = clases[2]

    return probabilidad_max, prediccion


def exactitud_precision_sensibilidad_fscore(TP, FN, FP, TN):
    """calcula la exactitud, precision, sensibilidad y f-score"""
    
    exactitud = (TP + TN) / (TP + FN + FP + TN)

    precision = TP / (TP + FP)

    sensibilidad = TP / (TP + FN)

    fscore = (2 * sensibilidad * precision) / (sensibilidad + precision)

    return exactitud, precision, sensibilidad, fscore


def matriz_confusion(clases, real_list, prediccion_list):
    """calcula e imprime la matriz de confusión"""
    
    matriz = [[sum([(real_list[i] == clase_real) and (prediccion_list[i] == clase_pred)
                    for i in range(len(real_list))])
               for clase_pred in clases]
              for clase_real in clases]
    matriz = np.stack(matriz)

    print('\nmatriz de confusion =\n', matriz)
    return matriz


def indices(clases, matriz):
    """obtiene los valores de la matriz de confusión para poder calcular la exactitud, precisión, sensibilidad y f-score y luego imprime los resultados"""
    exactitud_list = []
    precision_list = []
    sensibilidad_list = []
    fscore_list = []
    if len(clases) == 2:
        TP = matriz[0, 0]
        TN = matriz[1, 1]
        FP = matriz[1, 0]
        FN = matriz[0, 1]

        exactitud, precision, sensibilidad, fscore = exactitud_precision_sensibilidad_fscore(
            TP, FN, FP, TN)

        print('\nexactitud = ', exactitud)
        print('\nprecision = ', precision)
        print('\nsensibilidad = ', sensibilidad)
        print('\nfscore = ', fscore)

    elif len(clases) == 3:

        exactitud_parcial_list = []
        precision_parcial_list = []
        sensibilidad_parcial_list = []
        fscore_parcial_list = []

        a = matriz[0, 0]
        b = matriz[0, 1]
        c = matriz[0, 2]
        d = matriz[1, 0]
        e = matriz[1, 1]
        f = matriz[1, 2]
        g = matriz[2, 0]
        h = matriz[2, 1]
        i = matriz[2, 2]

        C1_TP = a
        C1_FN = b + c
        C1_FP = d + g
        C1_TN = e + f + h + i

        exactitud, precision, sensibilidad, fscore = exactitud_precision_sensibilidad_fscore(
            C1_TP, C1_FN, C1_FP, C1_TN)

        exactitud_parcial_list.append(exactitud)
        precision_parcial_list.append(precision)
        sensibilidad_parcial_list.append(sensibilidad)
        fscore_parcial_list.append(fscore)

        C2_TP = e
        C2_FN = d + f
        C2_FP = b + h
        C2_TN = a + c + g + i

        exactitud, precision, sensibilidad, fscore = exactitud_precision_sensibilidad_fscore(
            C2_TP, C2_FN, C2_FP, C2_TN)

        exactitud_parcial_list.append(exactitud)
        precision_parcial_list.append(precision)
        sensibilidad_parcial_list.append(sensibilidad)
        fscore_parcial_list.append(fscore)

        C3_TP = i
        C3_FN = g + h
        C3_FP = c + f
        C3_TN = a + b + d + e

        exactitud, precision, sensibilidad, fscore = exactitud_precision_sensibilidad_fscore(
            C3_TP, C3_FN, C3_FP, C3_TN)
        
        exactitud_parcial_list.append(exactitud)
        precision_parcial_list.append(precision)
        sensibilidad_parcial_list.append(sensibilidad)
        fscore_parcial_list.append(fscore)

        print('\nexactitud = ', exactitud_parcial_list)
        print('\nprecision = ', precision_parcial_list)
        print('\nsensibilidad = ', sensibilidad_parcial_list)
        print('\nfscore = ', fscore_parcial_list)

        exactitud = sum(exactitud_parcial_list) / len(exactitud_parcial_list)
        precision = sum(precision_parcial_list) / len(precision_parcial_list)
        sensibilidad = sum(sensibilidad_parcial_list) / len(sensibilidad_parcial_list)
        fscore = sum(fscore_parcial_list) / len(fscore_parcial_list)

        print('\nexactitud promedio = ', exactitud)
        print('\nprecision promedio = ', precision)
        print('\nsensibilidad promedio = ', sensibilidad)
        print('\nfscore promedio = ', fscore)
        
    exactitud_list.append(exactitud)
    precision_list.append(precision)
    sensibilidad_list.append(sensibilidad)
    fscore_list.append(fscore)
    
    return exactitud_list, precision_list, sensibilidad_list, fscore_list

def resultado_final(matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist):
    print('\n.................................................')
    print('\n................ resultado final ................')

    matriz_total = sum(matriz_masterlist)
    print('\n matriz de confusión final=\n',matriz_total)
    
    exactitud_masterlist = sum(exactitud_masterlist, [])
    exactitud_total = sum(exactitud_masterlist) / len(exactitud_masterlist)
    print('\n exactitud final=',exactitud_total)
    
    precision_masterlist = sum(precision_masterlist, [])
    precision_total = sum(precision_masterlist) / len(precision_masterlist)
    print('\n precision final=',precision_total)
    
    sensibilidad_masterlist = sum(sensibilidad_masterlist, [])
    sensibilidad_total = sum(sensibilidad_masterlist) / len(sensibilidad_masterlist)
    print('\n sensibilidad final=',sensibilidad_total)
    
    fscore_masterlist = sum(fscore_masterlist, [])
    fscore_total = sum(fscore_masterlist) / len(fscore_masterlist)
    print('\n f-score final=',fscore_total)
    print('\n.................................................')
    print('\n.................................................')
    return matriz_total,exactitud_total,precision_total,sensibilidad_total,fscore_total

def validacion_cruzada(df):
    """divide aleatoriamente el conjunto de entrenamiento en 10 grupos de aproximadamente el mismo tamaño y realiza los cálculos para cada grupo"""
    
    fold = 10

    df_shuffled = df.sample(frac=1, random_state=1)
    df_shuffled = df_shuffled.reset_index()

    folds = np.array_split(df_shuffled, 10)
    
    prediccion_masterlist = []
    real_masterlist = []
    media_masterlist = []
    varianza_masterlist = []
    matriz_masterlist = []
    exactitud_masterlist = []
    precision_masterlist = []
    sensibilidad_masterlist = []
    fscore_masterlist = []
    
    for f in range(fold):
        print('\n.................... fold', f+1, '....................')
        test = pd.concat(folds[f:f+1])
        train = pd.concat(folds[0:f]+folds[f+1::])

        label, clases_separadas, clases = separar_clases(df, df_shuffled)

        prediccion_list = []
        real_list = test.iloc[:, -1].values.tolist()
        real_masterlist.append(real_list)

        for row in range(len(test)):

            probabilidad_list = []

            for key in clases:

                PXC_list = []
                media_list = []
                varianza_list = []

                for column in range(test.columns[-1]):

                    x = test.iloc[row][column]
                    train_column = train.iloc[:, column+1]

                    count_all = len(train[(train[train.columns[-1]] == key)])

                    PXC = densidad_probabilidad(
                        x, test, train, train_column, key, column, count_all, PXC_list, media_list, varianza_list)
                    PXC_list.append(PXC)

                probabilidad = bayes(PXC_list, count_all, train)
                
                probabilidad_list.append(probabilidad)
            
            probabilidad_max, prediccion = pertenencia(
                probabilidad_list, clases)

            prediccion_list.append(prediccion)
            print("probabilidad=",probabilidad_list)
        media_masterlist.append(media_list)
        varianza_masterlist.append(varianza_list)
        prediccion_masterlist.append(prediccion_list)
        
        print('\nmedia =\n', media_list)
        print('\nvarianza =\n', varianza_list)
        
        matriz = matriz_confusion(clases, real_list, prediccion_list)
        
        matriz_masterlist.append(matriz)
        
        exactitud_list, precision_list, sensibilidad_list, fscore_list = indices(clases, matriz)
        
        exactitud_masterlist.append(exactitud_list)
        precision_masterlist.append(precision_list)
        sensibilidad_masterlist.append(sensibilidad_list)
        fscore_masterlist.append(fscore_list)
    
    return matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist

