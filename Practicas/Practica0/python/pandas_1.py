# Exercise 1
import pandas as pd
import numpy as np

# platform with no first party
def noParty(dato, fstParty):
    dato_platform=(dato["Platform"])
    fp=fstParty["Platform"]
    
    a =(set(dato_platform)) # quit reps 
    b= (set(fp))
    A=[]
    B=[]
    # formatting everything
    for elem in a:
        e= elem.upper()
        A.append(e)
    for elem in b:
        e= elem.upper()
        B.append(e)
        
    A=set(A)
    B=set(B)
    
    # substract lists
    return A.difference(B)

def noSales(dato,first):
    NASales=dato["NA_Sales"]
    EUSales=dato["EU_Sales"]
    JPSales=dato["JP_Sales"]
    OtherSales= dato["Other_Sales"]
    DPlatform=dato["Platform"]
    FPlatform= first["Platform"]
    FParty=first["First_Party"]
    
    NoSales=[]
    for elem in range(len(DPlatform)):
        #filter no sales in any continent (could format the data in int but i just needed the 0, not decimals of it)
        if(NASales[elem]=="0" and EUSales[elem]=="0" and JPSales[elem]=="0" and OtherSales[elem]=="0" ):  
            NoSales.append(DPlatform[elem].upper())
    
    NoSales=list(set(NoSales))
    FPlatform=list(set(FPlatform))
    NPlatform=[]
    
    # filter the fst party of the platforms with no sales (intersection between platform and nosales)
    for elem in range(len(FParty)):
        if(FPlatform[elem] in NoSales):
            NPlatform.append(FParty[elem])
            
    NPlatform=list(set(NPlatform)) # quit repetitions 
    
    return NPlatform


# MAIN

dato = pd.read_csv('dato.csv')
fstparty=pd.read_csv('first_party.csv')

nparty=noParty(dato,fstparty)
print(str(nparty))

nSales= noSales(dato,fstparty)
print(nSales)
