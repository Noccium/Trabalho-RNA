# <img style="float: right;" src="https://www.ufg.br/up/1/o/Marca_UFG_cor_completa_vertical.png" width=60px><br><br>
# <center> Trabalho Inteligência Computacional </center><br>
## <center> Redes Neurais Artificiais Reconhecimento de Gênero por Voz </center><br><br>
 <center> Lucas Felipe de Sousa Silva </center>
 <center> Matrícula - 201408050 </center><br><br>
 <center> Goiânia, 09 de julho de 2019  </center>

 <div style="page-break-after: always;"></div>

# Problema
Contruir uma MLP capaz de classificar o gênero de uma pessoa através das características da voz. Para isto foram utilizados dados disponibilizados no [Kaggle](https://www.kaggle.com/primaryobjects/voicegender).

- Classes: 2 (masculino, feminino)
- Características: 20
- Amostras: 3168

# Preparação dos dados
## Conjunto de treinamento
Foram utilizadas 3168 amostras sendo que 1584 destas para a classe masculino e 1584 para a classe feminino.

## Conjunto de testes
Foram utilizadas 1000 amostras sendo que 500 destas para a classe masculino e 500 para a classe feminino.

<div style="page-break-after: always;"></div>

## Conversões
Foi necessária a conversão do tipo de apenas uma coluna do dataset, label, que contém a classificação dos dados, "male" ou "female", de string para inteiro. As demais características permaneceram como tipo float.
- "male" = 1
- "female" = 0

# Arquitetura e Treinamento
- Quantidade de Camadas: 4
- Quantidade de Neurônios: 46
- Funções de ativação: elu, tanh, tanh, sigmoid
- Otimizador: Adagrad

| Camada | Qtd. Neurônios | Função de Ativação |
---|---|---
| 1 | 20 | elu
| 2 | 15 | tanh
| 3 | 10 | tanh
| 4 | 1 | sigmoid

<div style="page-break-after: always;"></div>

# Link Implementação
- https://github.com/Noccium/Trabalho-RNA
