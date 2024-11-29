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

sys.path.append("Funcoes_PACK")

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

# função que faz uma regressão ortogonal aos dados, retornando os parâmetros e
# intervalo de confiança.
# Fonte: http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/odr.html
# é uma regressão ortogonal, bastante propícia quando temos tanto incertezas no eixo
# x quanto no eixo y. Os CIs (intervalo para 95% de confiança já sai
# automaticamente).


def f(B, x):
    return B[0] * x + B[1]


def olsregress(X, Y):
    linear = odrpack.Model(f)
    mydata = odrpack.RealData(X, Y)
    myodr = odrpack.ODR(mydata, linear, beta0=[1.0, 2.0])
    myoutput = myodr.run()
    return myoutput.beta, myoutput.sd_beta


# função que calcula intervalos de confiança via bootstrap

import random
import numpy
import sklearn


def bootstrap(
    dataset, confidence=0.95, iterations=10000, sample_size=1.0, statistic=numpy.mean
):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.

    Args:
    dataset: A list of values, each a sample from an unknown population
    confidence: The confidence value (a float between 0 and 1.0)
    iterations: The number of iterations of resampling to perform
    sample_size: The sample size for each of the resampled (0 to 1.0
    for 0 to 100% of the original data size)
    statistic: The statistic to use. This must be a function that accepts
    a list of values and returns a single value.

    Returns:
    Returns the upper and lower values of the confidence interval.
    """
    stats = list()
    n_size = int(len(dataset) * sample_size)

    for _ in range(iterations):
        # Sample (with replacement) from the given dataset
        sample = sklearn.utils.resample(dataset, n_samples=n_size)
        # Calculate user-defined statistic and store it
        stat = statistic(sample)
        stats.append(stat)

    # Sort the array of per-sample statistics and cut off ends
    ostats = sorted(stats)
    lval = numpy.percentile(ostats, ((1 - confidence) / 2) * 100)
    uval = numpy.percentile(ostats, (confidence + ((1 - confidence) / 2)) * 100)

    return (lval, uval)


def space(data, num_of_points):

    if (data.ndim) > 1:
        mydata = data[0]
    else:
        mydata = data
    leng = mydata.size

    if num_of_points < 2:
        print("Escolha no minimo 2 janelas!")
        return None
    elif num_of_points > math.floor(leng / 2):
        print("O número de janelas é muito escolhido é muito grande!")
        return None
    else:
        minimo = mydata.min()
        maximo = mydata.max()
        temp = np.linspace(minimo, maximo, num=num_of_points + 1, endpoint=True)
        return temp

    return list()


def winmeanerr(data, num_of_points):
    # chama a função space: ok
    xs = space(data, num_of_points)

    if (data.ndim) != 2:
        print("O dado deve ter 2 colunas!")
        return None
    else:
        eixox = data[0]
        leng = list(range(len(eixox)))
        eixoy = data[1]

    # calcula o número de janelas com pelo menos um ponto dentro
    j = 0
    for i in range(num_of_points):
        tempx = np.extract(
            (eixox[leng] >= xs[i]) & (eixox[leng] < xs[i + 1]), eixox[leng]
        )
        if tempx.size > 0:
            j = j + 1
    tempx = np.extract(
        (eixox[leng] >= xs[num_of_points - 1]) & (eixox[leng] <= xs[num_of_points]),
        eixox[leng],
    )
    if tempx.size > 0:
        j = j + 1
    njanelas = j

    # cria vetores para armazenar as médias e erros
    medias = np.zeros((2, njanelas))
    xmin = np.zeros((njanelas, 1))
    xmax = np.zeros((njanelas, 1))
    ymin = np.zeros((njanelas, 1))
    ymax = np.zeros((njanelas, 1))

    j = 0
    for i in range(num_of_points):
        tempx = np.extract(
            (eixox[leng] >= xs[i]) & (eixox[leng] < xs[i + 1]), eixox[leng]
        )
        if tempx.size > 0:
            tempy = np.extract(
                (eixox[leng] >= xs[i]) & (eixox[leng] < xs[i + 1]), eixoy[leng]
            )
            medias[0][j] = np.mean(tempx)
            medias[1][j] = np.mean(tempy)
            xmin[j], xmax[j] = (
                pb.bootstrap(
                    tempx,
                    confidence=0.95,
                    iterations=1000,
                    sample_size=1.0,
                    statistic=np.mean,
                )
                - medias[0][j]
            )
            ymin[j], ymax[j] = (
                pb.bootstrap(
                    tempy,
                    confidence=0.95,
                    iterations=1000,
                    sample_size=1.0,
                    statistic=np.mean,
                )
                - medias[1][j]
            )
            j = j + 1

    tempx = np.extract(
        (eixox[leng] >= xs[num_of_points - 1]) & (eixox[leng] <= xs[num_of_points]),
        eixox[leng],
    )
    if tempx.size > 0:
        tempy = np.extract(
            (eixox[leng] >= xs[num_of_points - 1]) & (eixox[leng] <= xs[num_of_points]),
            eixoy[leng],
        )
        medias[0][njanelas - 1] = np.mean(tempx)
        medias[1][njanelas - 1] = np.mean(tempy)
        xmin[njanelas - 1], xmax[njanelas - 1] = (
            pb.bootstrap(
                tempx,
                confidence=0.95,
                iterations=1000,
                sample_size=1.0,
                statistic=np.mean,
            )
            - medias[0][njanelas - 1]
        )
        ymin[njanelas - 1], ymax[njanelas - 1] = (
            pb.bootstrap(
                tempy,
                confidence=0.95,
                iterations=1000,
                sample_size=1.0,
                statistic=np.mean,
            )
            - medias[1][njanelas - 1]
        )

    return medias[0], medias[1], [abs(xmin), abs(xmax)], [abs(ymin), abs(ymax)]
