import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

f1 = np.load("imagens-saida/f1.npy")  # shape (50, 512)
f2 = np.load("imagens-saida/f2.npy")  # shape (25, 512)
f3 = np.load("imagens-saida/f3.npy")  # shape (25, 512)

X = np.vstack([f1, f2, f3])  # concatena tudo
y = np.array([0]*len(f1) + [1]*len(f2) + [2]*len(f3))  # f1=0, f2=1, f3=2

svm = SVC(kernel="linear", C=1)  # separação com uma reta, gaussiano foi pior
# svm = SVC(kernel="rbf", C=1, gamma='scale')

n_splits = 5

seed = random.randint(0, 10000)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) # garante proporção, seed permanente

f1_folds = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1): # para cada fold de indice e testes dentro do range (X, y)
    X_train = X[train_idx] 
    X_test  = X[test_idx] 

    y_train = y[train_idx] # rótulos de classe
    y_test  = y[test_idx] 

    scaler = StandardScaler() # z-score, média 0 desvio padrãso 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm.fit(X_train, y_train) # treino
    y_pred = svm.predict(X_test) # predição

    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_folds.append(f1)
    print(f"fold {fold} teve f1_score de {f1:.4f}")

media_f1 = np.mean(f1_folds)
desvio_f1 = np.std(f1_folds)

print(f"acurácia média: {np.mean(media_f1):.4f}")
print(f"desvio padrão: {np.std(f1_folds):.4f}")