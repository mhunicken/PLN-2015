Práctico 2
==========

Ejercicio 1: Corpus Ancora: Estadísticas de etiquetas POS
---------------------------------------------------------

Resultados de la ejecución de ``stats.py``::

    Number of sents: 17379
    Number of occurrences of words: 517268
    Number of words (vocabulary): 46482
    Number of tags (tags vocabulary): 48

    Most frequent tags:
    Tag: nc (nombre común)
        Frequency: 92002 (17.79%)
        Most frequent words for tag:  años presidente millones equipo partido
    Tag: sp (preposición)
        Frequency: 79904 (15.45%)
        Most frequent words for tag:  de en a del con
    Tag: da (determinante artículo)
        Frequency: 54552 (10.55%)
        Most frequent words for tag:  la el los las El
    Tag: vm (verbo)
        Frequency: 50609 (9.78%)
        Most frequent words for tag:  está tiene dijo puede hace
    Tag: aq (adjetivo)
        Frequency: 33904 (6.55%)
        Most frequent words for tag:  pasado gran mayor nuevo próximo
    Tag: fc (coma)
        Frequency: 30148 (5.83%)
        Most frequent words for tag:  ,
    Tag: np (nombre propio)
        Frequency: 29113 (5.63%)
        Most frequent words for tag:  Gobierno España PP Barcelona Madrid
    Tag: fp (punto final)
        Frequency: 21157 (4.09%)
        Most frequent words for tag:  . ( )
    Tag: rg (adverbio)
        Frequency: 15333 (2.96%)
        Most frequent words for tag:  más hoy también ayer ya
    Tag: cc (conjunción copulativa)
        Frequency: 15023 (2.90%)
        Most frequent words for tag:  y pero o Pero e

    Levels of ambiguity:
    Level 1:
        Number of words: 44109 (94.89%)
        Most frequent words:  , el en con por
    Level 2:
        Number of words: 2194 (4.72%)
        Most frequent words:  la y " los del
    Level 3:
        Number of words: 153 (0.33%)
        Most frequent words:  . a un no es
    Level 4:
        Number of words: 19 (0.04%)
        Most frequent words:  de dos este tres todo
    Level 5:
        Number of words: 4 (0.01%)
        Most frequent words:  que mismo cinco medio
    Level 6:
        Number of words: 3 (0.01%)
        Most frequent words:  una como uno
    Level 7:
        Number of words: 0 (0.00%)
    Level 8:
        Number of words: 0 (0.00%)
    Level 9:
        Number of words: 0 (0.00%)

Ejercicio 2: Baseline Tagger
----------------------------
En ``__init__`` de ``BaselineTagger`` se guarda el tag más frecuente, y un diccionario con el tag más frecuente para cada palabra.

Ejercicio 3: Entrenamiento y Evaluación de Taggers
--------------------------------------------------
El script ``train.py`` provisto sólo se modificó para soportar todos los taggers implementados.
El script ``eval.py`` muestra accuracy general, y para palabras conocidas y desconocidas. Muestra la matriz de confusión entera (excepto filas y columnas compuestas de 0s). Por ser muy grandes, no se muestran en el README, sino que los resultados completos para cada modelo están en el directorio ``results/``

Evaluación
++++++++++
::

    Accuracy: 89.03%
    Accuracy (known words): 95.35%
    Accuracy (unknown words): 31.80%

    real 5.65
    user 5.40
    sys 0.14

La performance del Baseline es muy mala, sobre todo para palabras desconocidas. Se observa en la matriz de confusión que la mayoría de los errores son al taguear palabras como ``nc`` (el tag más común).

Ejercicio 4: Hidden Markov Models y Algoritmo de Viterbi
--------------------------------------------------------
La implementación de ``HMM`` es directa. Usa a ``ViterbiTagger`` para calcular el tagging más probable.
Para implementar ``ViterbiTagger.tag`` se siguió la estructura sugerida por los tests para la programación dinámica. En base a la última fila de la tabla, se decide tomar el tagging que maximiza la probabilidad (multiplicando también por la probabilidad de generar el fin de oración a partir de los n-1 últimos tags)

Ejercicio 5: HMM POS Tagger
---------------------------
La clase ``MLHMM`` extiende a ``HMM``, para reutilizar los métodos ``tag_prob``, ``tag_log_prob``, ``prob``, ``log_prob`` y ``tag``. La implementación es muy parecida al modelo de n-gramas del práctico 1. Se guardan los counts de secuencias de tags de largo n y n-1, los counts para cada par (palabra, tag) y los counts de cada tag. Se utiliza la opción de addone-smoothing para la transición de tags. Si una palabra es desconocida, la probabilidad de generarla es la misma para cada tag (positiva).

Evaluación
++++++++++

n=1::

    Accuracy: 89.01%
    Accuracy (known words): 95.32%
    Accuracy (unknown words): 31.80%

    real 36.73
    user 36.26
    sys 0.29

n=2::

    Accuracy: 92.72%
    Accuracy (known words): 97.61%
    Accuracy (unknown words): 48.42%

    real 192.40
    user 191.90
    sys 0.21

n=3::

    Accuracy: 93.17%
    Accuracy (known words): 97.67%
    Accuracy (unknown words): 52.31%

    real 917.14
    user 915.16
    sys 0.29

n=4::

    Accuracy: 93.14%
    Accuracy (known words): 97.44%
    Accuracy (unknown words): 54.14%

    real 4712.65
    user 4696.28
    sys 2.42

Vemos que con n=1, se obtienen resultados similares al Baseline (de hecho, son el mismo algoritmo). La mejor precisión se consigue con n=3. Sin embargo, sigue siendo pobre para palabras desconocidas. Debido a la complejidad del algoritmo de Viterbi, el tiempo de evaluación crece exponencialmente con el n.


Ejercicio 6: Features para Etiquetado de Secuencias
---------------------------------------------------
La implementación de los features es muy directa. Se tomó como referencia el feature de ejemplo.

Ejercicio 7: Maximum Entropy Markov Models
------------------------------------------
Se utiliza un pipeline compuesto del vectorizer definido a partir de las features del ejercicio 6, y un classifier (seleccionable entre ``LogisticRegression``, ``MultinomialNB`` o ``LinearSVC`` mediante un parámetro de ``__init__``). El tagging de una oración se hace de modo greedy, eligiendo en cada paso el mejor tag en base a los tags anteriores.

Evaluación
++++++++++

Logistic regression (n=1)::

    Accuracy: 92.70%
    Accuracy (known words): 95.28%
    Accuracy (unknown words): 69.32%

    real 64.39
    user 60.91
    sys 0.56

Logistic regression (n=2)::

    Accuracy: 91.99%
    Accuracy (known words): 94.55%
    Accuracy (unknown words): 68.75%

    real 68.53
    user 66.80
    sys 0.50

Logistic regression (n=3)::

    Accuracy: 92.18%
    Accuracy (known words): 94.72%
    Accuracy (unknown words): 69.22%

    real 68.52
    user 66.67
    sys 0.46

Logistic regression (n=4)::

    Accuracy: 92.23%
    Accuracy (known words): 94.72%
    Accuracy (unknown words): 69.62%

    real 68.91
    user 67.42
    sys 0.42

Vemos que aunque tienen menos precisión que el mejor MLHMM, tienen considerablemente más precisión en palabras no vistas. Esto es porque consideran características de las palabras desconocidas (por ejemplo, es probable que una palabra en mayúscula sea un nombre propio, de hecho se ven considerablemente menos errores de palabras de tag np mal tagueadas). Son mucho más rápidos de evaluar que los MLHMM, aunque más lentos de entrenar.

SVM (n=1)::

    Accuracy: 94.39%
    Accuracy (known words): 97.04%
    Accuracy (unknown words): 70.34%

    real 51.75
    user 51.29
    sys 0.35

SVM (n=2)::

    Accuracy: 94.28%
    Accuracy (known words): 96.91%
    Accuracy (unknown words): 70.47%

    real 59.09
    user 58.59
    sys 0.39

SVM (n=3)::

    Accuracy: 94.39%
    Accuracy (known words): 96.94%
    Accuracy (unknown words): 71.29%

    real 62.63
    user 62.20
    sys 0.33

SVM (n=4)::

    Accuracy: 94.45%
    Accuracy (known words): 96.96%
    Accuracy (unknown words): 71.71%

    real 65.32
    user 64.88
    sys 0.30

Los resultados para Support Vector Machines son similares a los comentados de Logistic Regression, con la diferencia de que el SVC tiene considerablemente mejor performance. El SVC con n=4 es entre todos el que mejor precisión logra, tanto en general como para palabras desconocidas.

Naive Bayes (n=1)::

    Accuracy: 88.27%
    Accuracy (known words): 92.18%
    Accuracy (unknown words): 52.85%

    real 2014.51
    user 2008.88
    sys 0.71

Naive Bayes (n=2)::

    Accuracy: 80.94%
    Accuracy (known words): 85.28%
    Accuracy (unknown words): 41.62%

    real 2014.51
    user 2008.94
    sys 1.02

Naive Bayes (n=3)::

    Accuracy: 74.34%
    Accuracy (known words): 78.10%
    Accuracy (unknown words): 40.28%

    real 2006.24
    user 2001.95
    sys 0.64

Naive Bayes (n=4)::

    Accuracy: 69.80%
    Accuracy (known words): 72.96%
    Accuracy (unknown words): 41.21%

    real 2012.39
    user 2008.20
    sys 0.47

El clasificador Multinomial Naive Bayes muestra una pésima performance. Esto es debido a que la simplificación de independencia entre features es demasiado fuerte para el problema en cuestión (se está asumiendo, por ejemplo, que el hecho de que una palabra sea "gato" es independiente de que sea "pescado"). También se ve que la situación empeora a medida que se agregan más features. También demora mucho la evaluación; debido a la simplicidad del algoritmo, es posible que se deba a una limitación de scikit-learn.
