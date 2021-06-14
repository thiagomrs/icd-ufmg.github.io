#!/usr/bin/env python
# coding: utf-8

# # Lista 01 - Introdução e Revisão Numpy
# 
# [NumPy](http://numpy.org) é um pacote incrivelmente poderoso em Python, onipresente em qualquer projeto de ciência de dados. Possui forte integração com o [Pandas](http://pandas.pydata.org), outra ferramenta que iremos abordar na matéria. NumPy adiciona suporte para matrizes multidimensionais e funções matemáticas que permitem que você execute facilmente cálculos de álgebra linear. Este notebook será uma coleção de exemplos de álgebra linear computados usando NumPy. 
# 
# ## Numpy 
# 
# Para fazer uso de Numpy precisamos importar a biblioteca

# In[1]:


# -*- coding: utf8

import numpy as np


# Quando pensamos no lado prático de ciência de dados, um aspecto chave que ajuda na implementação de novos algoritmos é a vetorização. De forma simples, vetorização consiste do uso de tipos como **escalar**, **vetor** e **matriz** para realizar uma computação mais eficaz (em tempo de execução).
# 
# Uma matriz é uma coleção de valores, normalmente representada por uma grade 𝑚 × 𝑛, onde 𝑚 é o número de linhas e 𝑛 é o número de colunas. Os comprimentos das arestas 𝑚 e 𝑛 não precisam ser necessariamente diferentes. Se tivermos 𝑚 = 𝑛, chamamos isso de matriz quadrada. Um caso particularmente interessante de uma matriz é quando 𝑚 = 1 ou 𝑛 = 1. Nesse caso, temos um caso especial de uma matriz que chamamos de vetor. Embora haja um objeto de matriz em NumPy, faremos tudo usando matrizes NumPy porque elas podem ter dimensões maiores que 2. 
# 
# 1. **Escalar:** Um vetor de zero dimensões

# In[2]:


1


# 2. **Vetor:** Representa uma dimensão

# Abaixo vamos criar um vetor simples. Inicialmente, vamos criar uma lista.

# In[3]:


data_list = [3.5, 5, 2, 8, 4.2]


# Observe o tipo da mesma.

# In[4]:


type(data_list)


# Embora vetores e listas sejam parecidos, vetores Numpy são otimizados para operações de Álgebra Linear. Ciência de Dados faz bastante uso de tais operações, sendo este um dos motivos da dependência em Numpy.
# 
# Abaixo criamos um vetor.

# In[5]:


data = np.array(data_list)
print(data)
print(type(data))


# Observe como podemos somar o mesmo com um número. Não é possível fazer tal operação com listas.

# In[6]:


data + 7


# 3. **Matrizes:** Representam duas dimensões.

# In[7]:


X = np.array([[2, 4],
              [1, 3]])
X


# Podemos indexar as matrizes e os vetores.

# In[8]:


data[0]


# In[9]:


X[0, 1] # aqui é primeira linha, segunda coluna


# Podemos também criar vetores/matrizes de números aleatórios

# In[10]:


# help(np.random.randn)
X = np.random.randn(4, 3) # Gera números aleatórios de uma normal
print(X)


# ### Indexando
# 
# Pegando a primeira linha

# In[11]:


X[0] # observe que 0 é a linha 1, compare com o X[0, 1] de antes.


# In[12]:


X[1] # segunda


# In[13]:


X[2] # terceira


# Observe como todos os tipos retornados são `array`. Array é o nome genérico de Numpy para vetores e matrizes. 

# `X[:, c]` pega uma coluna

# In[14]:


X[:, 0]


# In[15]:


X[:, 1]


# `X[um_vetor]` pega as linhas da matriz. `X[:, um_vetor]` pega as colunas

# In[16]:


X[[0, 0, 1]] # observe que pego a primeira linha, indexada por 0, duas vezes


# Abaixo pego a segunda a primeira coluna

# In[17]:


X[:, [1, 0]]


# ### Indexação Booleana
# 
# `X[vetor_booleano]` retorna as linhas (ou colunas quando X[:, vetor_booleano]) onde o vetor é true

# In[18]:


# X[[True, False, True, False]]
X[[True, True, True, True]]


# In[19]:


X[:, [False, True, True]]


# ### Reshape, Flatten e Ravel
# 
# Todo vetor ou matriz pode ser redimensionado. Observe como uma matriz abaixo de 9x8=72 elementos. Podemos redimensionar os mesmos para outros arrays de tamanho 72.

# In[20]:


X = np.random.randn(9, 8)


# Criando uma matriz de 18x4.

# In[21]:


X.reshape((18, 4))


# Ou um vetor de 72

# In[22]:


X.reshape(72)


# A chamada flatten e ravel faz a mesma coisa, criam uma visão de uma dimensão da matriz.

# In[23]:


X.flatten()


# In[24]:


X.ravel()


# As funções incorporadas ao NumPy podem ser facilmente chamadas em matrizes. A maioria das funções são aplicadas a um elemento de array (como a multiplicação escalar). Por exemplo, se chamarmos `log()` em um array, o logaritmo será obtido de cada elemento. 

# In[25]:


np.log(data)


# Mean tira a média

# In[26]:


np.mean(data)


# Algumas funções podem ser chamadas direto no vetor, nem todas serão assim. O importante é ler a [documentação](http://numpy.org) e aprender. Com um pouco de prática você vai se acostumando.

# In[27]:


data.mean()


# Abaixo temos a mediana,

# In[28]:


np.median(data) # por exemplo, não existe data.median(). Faz sentido? Não. Mas é assim.


# Em matrizes as funções operam em todos os elemntos.

# In[29]:


np.median(X)


# In[30]:


X.mean()


# In[31]:


np.log(X + 10)


# Porém, caso você queira a media de linhas ou colunas use `axis`. Antes, vamos ver o tamanho do vetor.

# In[32]:


X.shape


# In[33]:


np.mean(X, axis=0) # média das colunas. como temos 8 colunas, temos 8 elementos.


# In[34]:


np.mean(X, axis=0).shape


# In[35]:


np.mean(X, axis=1) # média das linhas


# In[36]:


np.mean(X, axis=1).shape


# Lembre-se que eixo 0 é coluna. Eixo 1 é linas.

# ### Multiplicação de Matrizes

# Para transpor uma matriz fazemos uso de .T

# In[37]:


X.shape


# In[38]:


X.T.shape


# In[39]:


X.T


# Para multiplicar matrizes, do ponto de visto de multiplicação matricial como definido na álgebra linear, fazemos uso de `@`.

# In[40]:


X @ X.T


# O uso de `*` realiza uma operação ponto a ponto

# In[41]:


X * X


# Observe a diferença de tamanhos

# In[42]:


(X * X).shape


# In[43]:


(X @ X.T).shape


# **Pense:** Para o nosso `X` de tamanho `(9, 8)`, qual o motivo de `X * X.T` não funcionar? Qual o motivo de `X @ X` não funcionar?

# ## Correção Automática
# 
# Nossa correção automática depende das funções abaixo. Tais funções comparam valores que serão computados pelo seu código com uma saída esperada. Normalmente, vocês não fazer uso de tais funções em notebooks como este. Porém, elas são chave em ambientes de testes automáticos (fora do nosso escopo).
# 
# Observe como algumas funções comparam valores e outras comparam vetores. Além do mais, temos funções para comparar dentro de algumas casas decimais.

# In[44]:


from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


# In[45]:


# caso você mude um dos valores vamos receber um erro!
assert_array_equal(2, 2)

# caso você mude um dos valores vamos receber um erro!
assert_array_equal([1, 2], [1, 2])

# caso você mude um dos valores vamos receber um erro!
assert_almost_equal(3.1415, 3.14, 1)


# Caso você mude um dos valores abaixo vamos receber um erro! Como o abaixo.
# 
# ```
# -----------------------------------------------------------------------
# AssertionError                        Traceback (most recent call last)
# <ipython-input-10-396672d880f2> in <module>
# ----> 1 assert_equal(2, 3) # caso você mude um dos valores vamos receber um erro!
# 
# ~/miniconda3/lib/python3.7/site-packages/numpy/testing/_private/utils.py in assert_equal(actual, desired, err_msg, verbose)
#     413         # Explicitly use __eq__ for comparison, gh-2552
#     414         if not (desired == actual):
# --> 415             raise AssertionError(msg)
#     416 
#     417     except (DeprecationWarning, FutureWarning) as e:
# 
# AssertionError: 
# Items are not equal:
#  ACTUAL: 2
#  DESIRED: 3
#  ```

# É essencial que todo seu código execute sem erros! Portanto, antes de submeter clique em `Kernel` no menu acima. Depois clique em `Restart & Execute All.`
# 
# **Garanta que o notebook executa até o fim!** Isto é, sem erros como o acima.

# ## Funções em Python

# Para criar uma função em Python fazemos uso da palavra-chave: 
# ```python
# def
# ```
# 
# Todos nossos exercícios farão uso de funções. **Mantenha a assinatura das funções exatamente como requisitado, a correção automática depende disso.** Abaixo, temos um exempo de uma função que imprime algo na tela!

# In[46]:


def print_something(txt):
    print(f'Você passou o argumento: {txt}')


# In[47]:


print_something('DCC 212')


# Podemos também dizer o tipo do argumento, porém faremos pouco uso disto em ICD.

# In[48]:


def print_something(txt: str):
    print(f'Você passou o argumento: {txt}')


# In[49]:


print_something('DCC 212')


# Abaixo temos uma função que soma, a soma, dois vetores

# In[50]:


def sum_of_sum_vectors(array_1, array_2):
    return (array_1 + array_2).sum()


# In[51]:


x = np.array([1, 2])
y = np.array([1, 2])


# In[52]:


sum_of_sum_vectors(x, y)


# Abaixo temos um teste, tais testes vão avaliar o seu código. Nem todos estão aqui no notebook!

# In[53]:


assert_equal(6, sum_of_sum_vectors(x, y))


# ## Exercício 01
# 
# Inicialmente, crie uma função que recebe duas listas de numéros, converte as duas para um vetor numpy usando `np.array` e retorna o produto interno das duas listas. 
# 
# __Dicas:__  
# 1. Tente fazer um código sem nenhum **for**! Ou seja, numpy permite operações em vetores e matrizes, onde: `np.array([1, 2]) + np.array([2, 2]) = np.array([3, 4])`.
# 
# __Funções:__
# 1. `np.sum(array)` soma os elementos do array. `array.sum()` tem o mesmo efeito!

# In[54]:


def inner(array_1, array_2):
   return np.inner(np.array(array_1), np.array(array_2))


# In[55]:


x1 = np.array([2, 4, 8])
x2 = np.array([10, 100, 1000])
assert_equal(20 + 400 + 8000, inner(x1, x2))


# ## Exercício 02
# 
# Implemente uma função utilizando numpy que recebe duas matrizes, multiplica as duas e retorne o valor médio das células da multiplicação. Por exemplo, ao multiplicar:
# 
# ```
# [1 2]
# [3 4] 
# 
# com 
# 
# [2 1]
# [1 2]
# 
# temos
# 
# [4  5 ]
# [10 11]
# 
# onde a média de [4, 5, 10, 11] é
# 
# 7.5, sua resposta final!
# ```
# 
# 
# __Dicas:__  
# 1. Use o operador @ para multiplicar matrizes!

# In[56]:


def medmult(X_1, X_2):
    x_y = X_1 @ X_2
    return np.mean(x_y)


# In[57]:


X = np.array([1, 2, 3, 4]).reshape(2, 2)
Y = np.array([2, 1, 1, 2]).reshape(2, 2)
assert_equal(7.5, medmult(X, Y))


# In[ ]:




