import argparse
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from svm2 import inteligencia_artificial

def interactive_mode():
    while True:
        print("\no que deseja fazer?")
        print("[1] descobrir melhores parâmetros - grid search")
        print("[2] treinar classificador")
        print("[3] testar em um dataset novo (pós treino)")

        choice2 = input("escolha uma opção: ").strip()

        if choice2 == "1":
            arq1 = input("\ninforme o nome do primeiro arquivo .npy de features: ").strip()
            arq2 = input("informe o nome do segundo arquivo .npy de features: ").strip()
            arq3 = input("informe o nome do terceiro arquivo .npy de features: ").strip()
            grid = inteligencia_artificial(arq1, arq2, arq3)
            grid.grid_search()

            finalizar = input("\nexecutar outro método?\n[S] sim\n[N] não\natenção: é case-sensitive\n").strip()
            if finalizar == "N":
                break
        elif choice2 == "2":
            arq1 = input("\ncaminho do arquivo .npy da classe 1: ").strip()
            arq2 = input("caminho do arquivo .npy da classe 2: ").strip()
            arq3 = input("caminho do arquivo .npy da classe 3: ").strip()
            dir_imagens = input("caminho base das imagens (se vazio, retorna índice): ").strip() or None
            kernel = input("kernel (ex: linear, rbf, poly): ").strip()
            c = float(input("margem de erro C: "))
            folds = int(input("número de folds: "))

            ia = inteligencia_artificial(arq1, arq2, arq3, kernel, c, folds, dir_imagens)
            ia.carregar_dados()
            ia.set_svm()
            ia.avaliar_modelo()

            finalizar = input("\nexecutar outro método?\n[S] sim\n[N] não\natenção: é case-sensitive\n").strip()
            if finalizar == "N":
                break
        elif choice2 == "3":
            arq1 = input("\ncaminho do arquivo .npy da classe 1: ").strip()
            arq2 = input("caminho do arquivo .npy da classe 2: ").strip()
            arq3 = input("caminho do arquivo .npy da classe 3: ").strip()
            dir_imagens = input("caminho base das imagens (se vazio, retorna índice): ").strip() or None
            kernel = input("kernel (ex: linear, rbf, poly): ").strip()
            c = float(input("margem de erro C: "))
            folds = int(input("número de folds: "))

            ia = inteligencia_artificial(arq1, arq2, arq3, kernel, c, folds, dir_imagens)
            ia.carregar_dados()
            ia.set_svm()
            ia.avaliar_modelo()

            ia.X = ia.scaler.fit_transform(ia.X)  # treinando em cima de todos os dados
            ia.svm.fit(ia.X, ia.y)

            n_arq1 = input("\ninforme o nome do NOVO primeiro arquivo .npy de features a ser TESTADO: ").strip()
            n_arq2 = input("informe o nome do NOVO segundo arquivo .npy de features a ser TESTADO: ").strip()
            n_arq3 = input("informe o nome do NOVO terceiro arquivo .npy de features a ser TESTADO: ").strip()
            f1_novo = ia.prever_novo_dataset(n_arq1, n_arq2, n_arq3)
            print(f"\nF1-score no novo dataset: {f1_novo:.4f}")

            finalizar = input("\nexecutar outro método?\n[S] sim\n[N] não\natenção: é case-sensitive\n").strip()
            if finalizar == "N":
                break