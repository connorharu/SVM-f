import argparse
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

class inteligencia_artificial:
    def __init__(self, arq1, arq2, arq3, kernel="linear", C=1, n_splits=5):
        self.scaler = StandardScaler()
        self.arq1 = arq1
        self.arq2 = arq2
        self.arq3 = arq3
        self.kernel = kernel
        self.C = C
        self.n_splits = n_splits
        self.svm = None
        self.X = None
        self.y = None

    def carregar_dados(self):
        f1 = np.load(self.arq1)
        f2 = np.load(self.arq2)
        f3 = np.load(self.arq3)

        self.X = np.vstack([f1, f2, f3])  # concatena tudo
        self.y = np.array([0]*len(f1) + [1]*len(f2) + [2]*len(f3))  # f1=0, f2=1, f3=2

    def grid_search(self):
        if self.X is None or self.y is None:
            if not (self.arq1 and self.arq2 and self.arq3):
                raise ValueError("os arquivos .npy devem ser informados para executar o grid search.")
            self.carregar_dados()

        graus = [0, 1, 5] # caso kernel polinomial, controla complexidade
        cs = [0.1, 1, 10] # margem de erro
        gammas = [0.1, 1, 10] # sensibilidade
        # kernels = ['linear', 'rbf', 'poly'] # separação entre classes (reta, curvada, muito curvada)
        param_grid = [
            {'kernel': ['linear'], 'C': cs},
            {'kernel': ['poly'], 'C': cs, 'degree': graus, 'gamma': gammas},
            {'kernel': ['rbf'], 'C': cs, 'gamma': gammas}
        ]

        model = SVC()
        gridSearch = GridSearchCV(estimator=model,param_grid=param_grid,cv=5) # cross validation de 5
        gridSearch.fit(self.X, self.y)
        print(f'a melhor escolha de parâmetros é: {gridSearch.best_estimator_}')

    def set_svm(self):
        self.svm = SVC(kernel=self.kernel, C=self.C)

    def avaliar_modelo(self):
        seed = random.randint(0, 10000)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        f1_folds = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y), 1): # para cada fold de indice e testes dentro do range (X, y)
            X_train = self.X[train_idx] 
            X_test  = self.X[test_idx] 

            y_train = self.y[train_idx] # rótulos de classe
            y_test  = self.y[test_idx] 

            scaler = self.scaler # z-score, média 0 desvio padrãso 1
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            self.svm.fit(X_train, y_train) # treino
            y_pred = self.svm.predict(X_test) # predição

            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_folds.append(f1)
            print(f"fold {fold} teve f1_score de {f1:.4f}")

        media_f1 = np.mean(f1_folds)
        desvio_f1 = np.std(f1_folds)

        print(f"acurácia média: {(media_f1):.4f}")
        print(f"desvio padrão: {np.std(f1_folds):.4f}")
        return media_f1, desvio_f1
    
    def prever_novo_dataset(self, arq1, arq2, arq3):
        f1_novo = np.load(arq1)
        f2_novo = np.load(arq2)
        f3_novo = np.load(arq3)

        X_novo = np.vstack([f1_novo, f2_novo, f3_novo])
        X_novo = self.scaler.transform(X_novo) # z-score
        y_novo = np.array([0]*len(f1_novo) + [1]*len(f2_novo) + [2]*len(f3_novo))

        y_pred_novo = self.svm.predict(X_novo)
        f1 = f1_score(y_novo, y_pred_novo, average='weighted')
        return f1

def main():
    parser = argparse.ArgumentParser(description="treino e validação de uma svm com cross-validation")
    subparsers = parser.add_subparsers(dest="command", help="método a ser executado")

    treinamento = subparsers.add_parser("treinamento", help="treinamento do classificador")
    treinamento.add_argument("--arq1", help="caminho do arquivo .npy da classe 1")
    treinamento.add_argument("--arq2", help="caminho do arquivo .npy da classe 2")
    treinamento.add_argument("--arq3", help="caminho do arquivo .npy da classe 3")
    treinamento.add_argument("--kernel", default="linear", help="kernel (ex: linear, rbf, poly)")
    treinamento.add_argument("--c", type=float, default=1.0, help="margem de erro C")
    treinamento.add_argument("--folds", type=int, default=5, help="número de folds")

    grid_search = subparsers.add_parser("grid_search", help="busca dos melhores parâmtetros para o classificador")
    grid_search.add_argument("--arq1", help="caminho do arquivo .npy da classe 1")
    grid_search.add_argument("--arq2", help="caminho do arquivo .npy da classe 2")
    grid_search.add_argument("--arq3", help="caminho do arquivo .npy da classe 3")

    dataset_teste = subparsers.add_parser("dataset_teste", help="teste do classificador com um novo dataset")
    dataset_teste.add_argument("--arq1", help="caminho do arquivo .npy da classe 1")
    dataset_teste.add_argument("--arq2", help="caminho do arquivo .npy da classe 2")
    dataset_teste.add_argument("--arq3", help="caminho do arquivo .npy da classe 3")
    dataset_teste.add_argument("--kernel", default="linear", help="kernel (ex: linear, rbf, poly)")
    dataset_teste.add_argument("--c", type=float, default=1.0, help="margem de erro C")
    dataset_teste.add_argument("--folds", type=int, default=5, help="número de folds")
    dataset_teste.add_argument("--novo_arq1", help="caminho do arquivo .npy da classe 1 a ser TESTADA")
    dataset_teste.add_argument("--novo_arq2", help="caminho do arquivo .npy da classe 2 a ser TESTADA")
    dataset_teste.add_argument("--novo_arq3", help="caminho do arquivo .npy da classe 3 a ser TESTADA")

    args = parser.parse_args()

    print("\ngostaria de executar passo a passo, ou sem auxílio?: ")
    print("[1] passo-a-passo")
    print("[2] sem auxílio")

    choice = input("escolha uma opção: ").strip()

    if choice == "1":
        from svm_interativo import interactive_mode
        interactive_mode()
        return
    elif choice == "2":
        print("\nexecutando o método pedido...")
    
    # metadados básicos
    if args.command == "treinamento":
        ia = inteligencia_artificial(args.arq1, args.arq2, args.arq3, args.kernel, args.c, args.folds)
        ia.carregar_dados()
        ia.set_svm()
        ia.avaliar_modelo()
    elif args.command == "grid_search":
        ia = inteligencia_artificial(args.arq1, args.arq2, args.arq3)
        ia.grid_search()
    elif args.command == "dataset_teste":
        ia = inteligencia_artificial(args.arq1, args.arq2, args.arq3, args.kernel, args.c, args.folds)
        ia.carregar_dados()
        ia.set_svm()
        ia.avaliar_modelo()

        ia.X = ia.scaler.fit_transform(ia.X)  # treinando em cima de todos os dados
        ia.svm.fit(ia.X, ia.y)
        f1_novo = ia.prever_novo_dataset(args.novo_arq1, args.novo_arq2, args.novo_arq3)
        print(f"\nF1-score no novo dataset: {f1_novo:.4f}")

if __name__ == "__main__":
    main()
