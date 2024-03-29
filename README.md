# Trabalho - Redes Neurais Artificiais
Este trabalho tem como objetivo desenvolver uma rede neural artificial que seja capaz de classificar o gênero de uma pessoa a partir das características de sua voz como frequência.

## voice.csv
O arquivo **voice.csv**, disponibilizado no [Kaggle](https://www.kaggle.com/primaryobjects/voicegender), contém os dados utilizados para o treinamento da rede. A tabela possui 21 colunas e 3168 linhas dispostas da seguinte forma.

- **meanfreq**: mean frequency (in kHz)
- **sd**: standard deviation of frequency
- **median**: median frequency (in kHz)
- **Q25**: first quantile (in kHz)
- **Q75**: third quantile (in kHz)
- **IQR**: interquantile range (in kHz)
- **skew**: skewness (see note in specprop description)
- **kurt**: kurtosis (see note in specprop description)
- **sp.ent**: spectral entropy
- **sfm**: spectral flatness
- **mode**: mode frequency
- **centroid**: frequency centroid (see specprop)
- **peakf**: peak frequency (frequency with highest energy)
- **meanfun**: average of fundamental frequency measured across acoustic signal
- **minfun**: minimum fundamental frequency measured across acoustic signal
- **maxfun**: maximum fundamental frequency measured across acoustic signal
- **meandom**: average of dominant frequency measured across acoustic signal
- **mindom**: minimum of dominant frequency measured across acoustic signal
- **maxdom**: maximum of dominant frequency measured across acoustic signal
- **dfrange**: range of dominant frequency measured across acoustic signal
- **modindx**: modulation index. Calculated as the accumulated absolute  difference between adjacent measurements of fundamental frequencies
divided by the frequency range
- **label**: male or female


## rna.py
Arquivo executável do treinamento da rede neural artificial.

## model.py
Arquivo executável do modelo da rede neural artificial de melhor acurácia obtido a partir de seu treinamento.

## my_model.h5
Arquivo do modelo da rede neural artificial de melhor acurácia obtido a partir de seu treinamento.
