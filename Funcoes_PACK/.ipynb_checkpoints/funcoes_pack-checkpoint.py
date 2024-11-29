"""

PACOTE DE FUNÇÕES COMPARTILHADAS.
SEM LICENÇA NO MOMENTO

"""


# ---------------------------------------------------------------------------- #
# função para teste

def printa(myText):
	print(myText)

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# importando os pacotes necessários para as funções abaixo

import sys
sys.path.append('Funcoes_PACK')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import stats
import codecs
import scipy.odr.odrpack as odrpack
import math
import pandas as pd
import seaborn as sns
import pybootstrap_mais_mais as pb


# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# função que divide o dado em janelas equidistantes

def space(data, num_of_points):
	
	if (data.ndim) > 1:
		mydata = np.transpose(data)[0]
	else:
		mydata = data
	
	if num_of_points > 1:
		minimo = mydata.min()
		maximo = mydata.max()
		temp = np.linspace(minimo, maximo, num=num_of_points+1, endpoint=True)
		return temp
	else:
		print("Num_of_points invalido!")
	
	return list()
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# função que divide o dado em janelas log-espaçadas (no eixo x)

def logspace(data, num_of_points):
	
	if (data.ndim) > 1:
		mydata = np.log10(np.transpose(data)[0])
	else:
		mydata = np.log10(data)
	
	if num_of_points > 1:
		minimo = mydata.min()
		maximo = mydata.max()
		temp = np.logspace(minimo, maximo, num=num_of_points+1, base=10.0, endpoint=True)
		return temp
	else:
		print("Num_of_points invalido!")
	
	return list()
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# função que calcula as médias dentro das janelas equidistantes e retorna 2 vetores: eixox, eixoy

def winmean(data, num_of_points):
	if (data.ndim) != 2:
		print("O dado deve ter 2 colunas!")
	else:
		eixox = np.transpose(data)[0]
		leng = list( range( len(eixox) ) )
		eixoy = np.transpose(data)[1]

	# chama a função logspace: ok
	xs = space(eixox, num_of_points)
	
	medias = np.zeros((2, num_of_points))
	
	for i in range(num_of_points-1):
		medias[0][i] = np.mean(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixox[leng]))
		medias[1][i] = np.mean(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixoy[leng]))
	
	medias[0][num_of_points-1] = np.mean(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixox[leng]))
	medias[1][num_of_points-1] = np.mean(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixoy[leng]))
	
	return medias[0], medias[1]
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# função que calcula as médias dentro das janelas log-espaçadas e retorna 2 vetores: eixox, eixoy

def winlogmean(data, num_of_points):
	if (data.ndim) != 2:
		print("O dado deve ter 2 colunas!")
	else:
		eixox = np.transpose(data)[0]
		leng = list( range( len(eixox) ) )
		eixoy = np.transpose(data)[1]

	# chama a função logspace: ok
	xs = logspace(eixox, num_of_points)
	
	medias = np.zeros((2, num_of_points))
	
	for i in range(num_of_points-1):
		medias[0][i] = np.mean(np.log10(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixox[leng])))
		medias[1][i] = np.mean(np.log10(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixoy[leng])))
	
	medias[0][num_of_points-1] = np.mean(np.log10(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixox[leng])))
	medias[1][num_of_points-1] = np.mean(np.log10(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixoy[leng])))
	
	return np.power(10, medias[0]), np.power(10, medias[1])
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# função que calcula as médias dentro das janelas equidistantes e retorna 4 vetores: eixox, eixoy, xerr, yerr

def winmeanerr(data, num_of_points):
	if (data.ndim) != 2:
		print("O dado deve ter 2 colunas!")
	else:
		eixox = np.transpose(data)[0]
		leng = list( range( len(eixox) ) )
		eixoy = np.transpose(data)[1]

	# chama a função logspace: ok
	xs = space(eixox, num_of_points)
	
	medias = np.zeros((2, num_of_points))
	xmin = np.zeros((num_of_points,1))
	xmax = np.zeros((num_of_points,1))
	ymin = np.zeros((num_of_points,1))
	ymax = np.zeros((num_of_points,1))
	
	for i in range(num_of_points-1):
		tempx = np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixox[leng])
		tempy = np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixoy[leng])
		medias[0][i] = np.mean(tempx)
		medias[1][i] = np.mean(tempy)
		xmin[i], xmax[i] = pb.bootstrap(tempx, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[0][i]
		ymin[i], ymax[i] = pb.bootstrap(tempy, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[1][i]
	
	tempx = np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixox[leng])
	tempy = np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixoy[leng])
	medias[0][num_of_points-1] = np.mean(tempx)
	medias[1][num_of_points-1] = np.mean(tempy)
	xmin[num_of_points-1], xmax[num_of_points-1] = pb.bootstrap(tempx, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[0][num_of_points-1]
	ymin[num_of_points-1], ymax[num_of_points-1] = pb.bootstrap(tempy, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[1][num_of_points-1]
	
	return medias[0], medias[1], [abs(xmin), abs(xmax)], [abs(ymin), abs(ymax)]
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# função que faz uma regressão ortogonal aos dados, retornando os parâmetros e intervalo de confiança

# Tirei essa função daqui:
# http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/odr.html
# é uma regressão ortogonal, bastante propícia quando temos tanto incertezas no eixo x quanto no eixo y. Os CIs (intevalo para 95% de confiança) já é calculado automaticamente pela função Student T)


def f(B, x):
	return B[0]*x + B[1]

def olsregress(X,Y):
	# ppf = percent point function
	temp = stats.t(df=len(X)).ppf((0.025, 0.975))[1]

	linear = odrpack.Model(f)
	mydata = odrpack.RealData(X, Y)
	myodr = odrpack.ODR(mydata, linear, beta0=[1., 2.])
	myoutput = myodr.run()
	return myoutput.beta, temp*myoutput.sd_beta
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# função que calcula as médias dentro das janelas equidistantes e retorna 4 vetores: eixox, eixoy, xerr, yerr

def winlogmeanerr(data, num_of_points):
	if (data.ndim) != 2:
		print("O dado deve ter 2 colunas!")
	else:
		eixox = np.transpose(data)[0]
		leng = list( range( len(eixox) ) )
		eixoy = np.transpose(data)[1]
	
	# chama a função logspace: ok
	xs = fun.logspace(eixox, num_of_points)
	
	medias = np.zeros((2, num_of_points))
	xmin = np.zeros((num_of_points,1))
	xmax = np.zeros((num_of_points,1))
	ymin = np.zeros((num_of_points,1))
	ymax = np.zeros((num_of_points,1))
	
	for i in range(num_of_points-1):
		tempx=np.log10(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixox[leng]))
		tempy=np.log10(np.extract((eixox[leng] >= xs[i]) & (eixox[leng] < xs[i+1]), eixoy[leng]))
		medias[0][i] = np.mean(tempx)
		medias[1][i] = np.mean(tempy)
		xmin[i], xmax[i] = pb.bootstrap(tempx, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[0][i]
		ymin[i], ymax[i] = pb.bootstrap(tempy, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[1][i]
	
	tempx = np.log10(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixox[leng]))
	tempy = np.log10(np.extract((eixox[leng] >= xs[num_of_points-1]) & (eixox[leng] <= xs[num_of_points]), eixoy[leng]))
	medias[0][num_of_points-1] = np.mean(tempx)
	medias[1][num_of_points-1] = np.mean(tempy)
	xmin[num_of_points-1], xmax[num_of_points-1] = pb.bootstrap(tempx, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[0][num_of_points-1]
	ymin[num_of_points-1], ymax[num_of_points-1] = pb.bootstrap(tempy, confidence=0.95, iterations=1000, sample_size=1.0, statistic=np.mean) - medias[1][num_of_points-1]
	
	return np.power(10, medias[0]), np.power(10, medias[1]), np.power(10, [abs(xmin), abs(xmax)]), np.power(10, [abs(ymin), abs(ymax)])

# ---------------------------------------------------------------------------- #
# nova função aqui...
#regrassão linear por machine learning


# ---------------------------------------------------------------------------- #
# nova função aqui...
