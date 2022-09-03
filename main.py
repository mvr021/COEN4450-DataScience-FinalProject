# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:57:47 2022

@author: yeya
"""
from program import *


menu = {
    1:'manual',
    2:'automático',
    3:'salir'
}

menu_datos = {
    1:'data_1.csv',
    2:'data_2.csv',
    3:'data_3.csv',
    4:'data_4.csv'
}

def print_menu():
    """imprime el menú"""
    
    print('\nMenu:')
    for key in menu.keys():
        print (key, '--', menu[key])
        
def print_menu_datos():
    """imprime el submenú de datos"""
    
    print('\nSubmenu:')
    for key in menu_datos.keys():
        print (key, '--', menu_datos[key])

        
#Seleccionar y actuar        
flag = True

while flag == True:
    
    print_menu()
    opcion = int(input('Seleccione una opción: '))
   
    #Manual
    if opcion == 1:
        
        #selección de datos
        print_menu_datos()
        opcion_datos = int(input('Seleccione una opción: '))
        
        if opcion_datos > 4:
            print(('Opción inválida. Reiniciando...'))
            continue
        
        else:
            reduccion = input('reducir características? si/no: ')
            if reduccion == 'si':
                grafica = input('graficar? si/no: ')  
            datos = menu_datos[opcion_datos]
            df = pd.read_csv(datos,header=None)
            
            if reduccion == 'si':
               df = reducir(df,datos)
               if grafica == 'si':
                   label,clases_separadas,clases = separar_clases(df,df)
                   graficar(df,clases,datos)
            print("\n..................",datos,"..................")        
            matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist = validacion_cruzada(df)
            matriz_total,exactitud_total,precision_total,sensibilidad_total,fscore_total = resultado_final(matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist)
        
  
        
    #Automático
    elif opcion == 2:
        reduccion = input('reducir características? si/no: ')
        if reduccion == 'si':
            grafica = input('graficar? si/no: ')  

        for key in menu_datos.keys():
            print("\n..................",menu_datos[key],"..................")
            df = pd.read_csv(menu_datos[key],header=None)
            
            if reduccion == 'si':
                df = reducir(df,menu_datos[key])
              
                if grafica == 'si':
                   label,clases_separadas,clases = separar_clases(df,df)
                   graficar(df,clases,menu_datos[key])
                    
            matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist = validacion_cruzada(df)
            matriz_total,exactitud_total,precision_total,sensibilidad_total,fscore_total = resultado_final(matriz_masterlist,exactitud_masterlist,precision_masterlist,sensibilidad_masterlist,fscore_masterlist)

        
    #Salir
    elif opcion == 3:
        print('\nSaliendo...\n')
        flag = False
        
    else:
        print('Opción inválida. Ingrese un número entre 1 y 3.')

   