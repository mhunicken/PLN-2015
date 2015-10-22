Práctico 2
==========

Ejercicio 1: Corpus Ancora: Estadísticas de etiquetas POS
---------------------------------------------------------

Resultados de la ejecución de stats.py::

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

Ejercicio 4: Hidden Markov Models y Algoritmo de Viterbi
--------------------------------------------------------
La implementación de ``HMM`` es directa. Usa a ``ViterbiTagger`` para calcular el tagging más probable.
Para implementar ``ViterbiTagger.tag`` se siguió la estructura sugerida por los tests para la programación dinámica. En base a la última fila de la tabla, se decide tomar el tagging que maximiza la probabilidad (multiplicando también por la probabilidad de generar el fin de oración a partir de los n-1 últimos tags)

Ejercicio 5: HMM POS Tagger
---------------------------
La clase ``MLHMM`` extiende a ``HMM``, para reutilizar los métodos ``tag_prob``, ``tag_log_prob``, ``prob``, ``log_prob`` y ``tag``. La implementación es muy parecida al modelo de n-gramas del práctico 1. Se guardan los counts de secuencias de tags de largo n y n-1, los counts para cada par (palabra, tag) y los counts de cada tag. Se utiliza la opción de addone-smoothing para la transición de tags. Si una palabra es desconocida, la probabilidad de generarla es la misma para cada tag (positiva).

Ejercicio 6: Features para Etiquetado de Secuencias
---------------------------------------------------
La implementación de los features es muy directa. Se tomó como referencia el feature de ejemplo.

Ejercicio 7: Maximum Entropy Markov Models
------------------------------------------
TODO
