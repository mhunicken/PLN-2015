Práctico 3
==========

Ejercicio 1: Evaluación de Parsers
----------------------------------

Se extendió eval.py para evaluar sobre las primeras ``n`` oraciones, o sobre las oraciones de largo menor o igual a ``m``. También se calculan las métricas unlabeled.

Evaluación lbranch
++++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 8.81%
      Recall: 14.57%
      F1: 10.98%
    Unlabeled
      Precision: 14.71%
      Recall: 24.33%
      F1: 18.33%

    real    0m5.645s
    user    0m5.585s
    sys     0m0.056s


Evaluación rbranch
++++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 8.81%
      Recall: 14.57%
      F1: 10.98%
    Unlabeled
      Precision: 8.87%
      Recall: 14.68%
      F1: 11.06%

    real    0m5.455s
    user    0m5.384s
    sys     0m0.068s


Evaluación flat
+++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 99.93%
      Recall: 14.57%
      F1: 25.43%
    Unlabeled
      Precision: 100.00%
      Recall: 14.58%
      F1: 25.45%

    real    0m5.069s
    user    0m5.005s
    sys     0m0.060s


Ejercicio 2: Algoritmo CKY
--------------------------
Para implementar ``CKYParser.parse`` se siguió la estructura sugerida por los tests para la programación dinámica.
Se toma el parsing que maximiza la probabilidad del símbolo inicial en el rango de la oración completa.
Como, debido al tipo de normalización que se realiza posteriormente en UPCFG, hay muchas reglas, y muchos tags que ocurren en pocas producciones, resulta más eficiente iterar sobre los no-terminales posibles de cada intervalo, en lugar de iterar sobre el total de las producciones. Se utiliza un diccionario que mapea cada par de no-terminales a el conjunto de reglas que los producen.

Ejercicio 3: PCFGs No Lexicalizadas
-----------------------------------
En ``UPCFG.__init__`` se construye una PCFG mediante counts sobre las producciones de las oraciones del training set, convertidas a CNF con los métodos ``chomsky_normal_form`` y ``collapse_unary`` de ``nltk.Tree``. Se inicializa también un ``CKYParser``.
En ``UPCFG.parse`` se parsea usando el ``CKYParser`` y se deshace la normalización con ``un_chomsky_normal_form``. En caso de que no se encuentre ningún parsing posible, se devuelve el parsing plano.

Evaluación
++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 73.25%
      Recall: 72.95%
      F1: 73.10%
    Unlabeled
      Precision: 75.36%
      Recall: 75.05%
      F1: 75.21%

    real    4m14.682s
    user    4m14.080s
    sys     0m0.432s


Ejercicio 4: Markovización Horizontal
-------------------------------------
Para extender UPCFG con markovización horizontal, sólo hubo que pasar el parámetro opcional horzMarkov a ``chomsky_normal_form`` al momento de realizar la normalización de los árboles del test set. De este modo, en las producciones de más de dos hijos, al normalizarlas, se consideran sólo los últimos n no-terminales de la produción para nombrar el nuevo no-terminal. Esto reduce la cantidad de no-terminales presentes en la PCFG.

Evaluación (n=0)
++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 70.25%
      Recall: 70.02%
      F1: 70.14%
    Unlabeled
      Precision: 72.11%
      Recall: 71.88%
      F1: 72.00%

    real    1m19.797s
    user    1m19.758s
    sys     0m0.088s


Evaluación (n=1)
++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 74.67%
      Recall: 74.58%
      F1: 74.62%
    Unlabeled
      Precision: 76.53%
      Recall: 76.43%
      F1: 76.48%

    real    1m57.089s
    user    1m56.753s
    sys     0m0.128s



Evaluación (n=2)
++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 74.89%
      Recall: 74.37%
      F1: 74.63%
    Unlabeled
      Precision: 76.81%
      Recall: 76.28%
      F1: 76.55%

    real    3m8.100s
    user    3m7.781s
    sys     0m0.196s


Evaluación (n=3)
++++++++++++++++
::

    Parsed 1444 sentences
    Labeled
      Precision: 74.10%
      Recall: 73.47%
      F1: 73.78%
    Unlabeled
      Precision: 76.26%
      Recall: 75.61%
      F1: 75.93%

    real    3m47.299s
    user    3m47.172s
    sys     0m0.260s


Evaluación UPCFG (resumen)
++++++++++++++++++++++++++

+-----+-----------------------+-----------------------+-----------------------+
|     | Precision             | Recall                | F1                    |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+
| n   | Labeled   | Unlabeled | Labeled   | Unlabeled | Labeled   | Unlabeled |
+=====+===========+===========+===========+===========+===========+===========+
| 0   | 70.25     | 72.11     | 70.02     | 71.88     | 70.14     | 72.00     |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+
| 1   | 74.67     | 76.53     | 74.58     | 76.43     | 74.62     | 76.48     |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+
| 2   | 74.89     | 76.81     | 74.37     | 76.28     | 74.63     | 76.55     |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+
| 3   | 74.10     | 76.26     | 73.47     | 75.61     | 73.78     | 75.93     |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+
| inf | 73.25     | 75.36     | 72.95     | 75.05     | 73.10     | 75.21     |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+

(n es el orden de markovización horizontal. "n=inf" se refiere a la UPCFG sin markovización).
Para todas las métricas, el mejor valor se alcanza en n=2. Valores más altos de n (o no markovización) hacen overfitting del training set.
Aplicar markovización horizontal tiene también la ventaja de que se disminuyen los tiempos de evaluación, debido a que hay menos reglas y no-terminales.
