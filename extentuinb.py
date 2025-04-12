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