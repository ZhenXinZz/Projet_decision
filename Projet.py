import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
exc = pd.ExcelFile("PaysIN.xlsx")
df = pd.read_excel(exc)
data = df[1:].values
print(type(data))

valeur = data[:, 1:]

def Pareto_domine(X,Y):
    sup_strict=False
    for i in range(len(X)):
        if X[i]<Y[i]:
            return False
        if X[i]>Y[i]:
            sup_strict=True
    return sup_strict

def Pareto_domine_ver2(X,Y):
    sup_strict=False
    indice=[2,8,9]
    for i in indice:
        if X[i]<Y[i]:
            return False
        if X[i]>Y[i]:
            sup_strict=True
    return sup_strict


nb_total=0
nb_verifiee=0
for i in range (len (valeur)):
    for j in range (len(valeur)):
        if i!=j:
            nb_total+=1
            if Pareto_domine(valeur[i],valeur[j]):
                nb_verifiee+=1
print(nb_verifiee)
print(nb_total)
print(nb_verifiee/nb_total)
            
nb_total=0
nb_verifiee=0
for i in range (len (valeur)):
    for j in range (len(valeur)):
        if i!=j:
            nb_total+=1
            if Pareto_domine_ver2(valeur[i],valeur[j]):
                nb_verifiee+=1
print(nb_verifiee)
print(nb_total)
print(nb_verifiee/nb_total)
            

def Irreflexive (relation):
    for i in valeur:
        if relation(i,i) == True:
            return False
    return True

def Non_Symetrique(relation):
    for i in valeur:
        for j in valeur:
            if relation(i,j)==True:
                if relation(j,i)==False:
                    return True
    return False

def Antisymetrique(relation):
    for i in valeur:
        for j in valeur:
            if relation(i,j)==True:
                if relation(j,i)==True:
                    if i!=j:
                        return False
    return True

def Asymetrique (relation):
    for i in valeur:
        for j in valeur:
            if relation(i,j)==True:
                if relation(j,i)==True:
                    return False
    return True

def transitive(relation):
    for i in valeur:
        for j in valeur:
            for x in valeur:
                if relation(i,x)== True and relation(x,j)== True:
                    if relation(i,j)==False:
                        return False
    return True
'''

def negativement_transitive(relation):
    for i in valeur:
        for j in veleur:
            for x in valeur:
                if relation(i,j)==True:
                    if relation(i,x) ==False and relation(x,j)==False:
                        return False
    return True

def complet(relation):
    n=len(valeur)
    for i in range (n):
        for j in range(i+1,n):
            if relation(i,j)==False and relation(j,i)==False:
                return False
    return True
'''
'''
data = np.column_stack((valeur[:, 0], valeur[:, 2], valeur[:, 4]))
print(data)
def plot_pareto_front(df, x,y,x_col, y_col):
    pareto = []
    for i in range(len(data)):
        dominated = False
        for j in range(len(data)):
            if i != j and Pareto_domine(data[j], data[i]):
                dominated = True
                break
        if not dominated:
            pareto_points.append(data[i])
    plt.scatter(df[:,x], df[:,y], c='blue', label="Pays")
    plt.scatter(pareto[:, x], pareto[:, y], color="red", label="帕累托前沿")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid()
    plt.show()

plot_pareto_front(df, "CO2", "Dépendance_énergétique")

'''





'''
nb=0
n=len(valeur)
for i in range (n):
        for j in range(i+1,n):
            if Pareto_domine(i,j)==True or Pareto_domine(j,i)==True:
                nb+=1
print (nb)

'''
from mip import Model, MINIMIZE, INTEGER,CONTINUOUS,xsum

def L1_inv(X,Y):
    n=len(X)
    m=Model(sense=MINIMIZE)
    poids=[1/n]*n
    w=[m.add_var(name=('f'+str(i)),lb=0,ub=1,var_type=CONTINUOUS) for i in range(n)]
    z=[m.add_var(name=('f'+str(i)),lb=0,var_type=CONTINUOUS) for i in range(n)]
    m+=xsum(w[i] for i in range(n)) == 1
    m += xsum(w[i] * X[i] for i in range(n)) <= xsum(w[i] * Y[i] for i in range(n))
    for i in range(n):
        m += z[i] >= w[i] - poids[i]
        m += z[i] >= poids[i] - w[i]
    m.objective = xsum(z)
    m.optimize()
    return m.objective_value
    
print(L1_inv(valeur[0], valeur[1]))

n=len(valeur[0])
L1_matrix = np.zeros((n, n))
for i in range (n):
    for j in range (n):
        if i!=j:
            L1_matrix[i, j] = L1_inv(valeur[i], valeur[j])
         
x_vals = np.linspace(0, 2, 100)
y_vals = [sum(l1 <= x for l1 in L1_matrix) for x in x_vals]

# 画图
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Nombre de paires (X,Y) avec L1_inv ≤ x")
plt.xlabel("Seuil x sur L1_inv")
plt.ylabel("Nombre de paires (X, Y)")
plt.title("Distribution cumulative des distances L1_inv")
plt.grid(True)
plt.legend()
plt.show()


def MGP(X,w):
    produit=1
    for i in range(11):
        produit*=X[i]**w[i]
    produit=produit**(1/sum(w))
    return produit

def Wald(X):
    return min(X)

def Hurwicz(X,a):
    return a*min(X)+(1-a)*max(X)

w1=[1/11 for i in range(11)]
w2=[1 for i in range (11)]
w3=[0 for i in range (10)]+[1]

print(MGP(valeur[0],w1))
print(MGP(valeur[0],w2))
print(MGP(valeur[0],w3))

print(Hurwicz(valeur[0],0))
print(Hurwicz(valeur[0],1))
print(Hurwicz(valeur[0],0.5))




'''
def Kendall_Tau(R,w,Q,a):
    if w=None:
        for i in range (len(Valeur)):
            for j in range (len(Valeur)):
                if a==None:
                    
def Electre(X,Y):
    nb=0
    for i in range (len(X)):
        if X[1]>=Y[1]:
            nb+=1
        if Y[i]-X[i]>=1.5:
            return False
    if nb<7:
        return False
    return True

A=[8 for i in range (len(Valeur[1]))]
B=[6 for i in range (len(Valeur[1]))]
C=[4 for i in range (len(Valeur[1]))]
D=[2 for i in range (len(Valeur[1]))]

categorie_A=[]
categorie_B=[]
categorie_C=[]
categorie_D=[]
categorie_E=[]


def Categorie(X):
    pays=X[0]
    Y=X[1:]
    if Electre(Y,A)==True:
        categorie_A.append(pays)
        return None
    elif Electre(Y,B)==True:
        categorie_B.append(pays)
        return None
    elif Electre(Y,C)==True:
        categorie_C.append(pays)
        return None
    elif Electre(Y,D)==True:
        categorie_D.append(pays)
        return None
    else :
        categorie_E.append(pays)
        reture None
'''
