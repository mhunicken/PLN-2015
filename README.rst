PLN 2015: Procesamiento de Lenguaje Natural 2015
================================================

Ejercicio 1
-----------
Se utiliza como corpus una pequeña porción del dump oficial de la Wikipedia en español, preprocesado para eliminarle subtítulos y tags de inicio y fin de artículo. Se utiliza como train spanishText_20000_25000_small (~8MB) y como test spanishText_480000_485000_small (~1MB). Se utiliza como corpus reader PlainTextCorpusReader con el tokenizer por defecto.

Ejercicio 2
-----------
Se utilizó como base la clase NGram provista. Antes de procesar cada oración se le agregan n-1 tags de inicio y el tag de finalización.

Ejercicio 3
-----------
En la inicialización de NgramGenerator se recorre el diccionario counts del modelo, y en base al mismo se inicializan los dos diccionarios probs y sorted_probs (probs ordenadas de mayor a menor). generate_token genera un token en base a las probabilidades dadas por sorted_probs. generate_sent genera una oración generando con (generate_token) un token en base a los anteriores hasta llegar al fin de oración.

Ejemplos de oraciones generadas
+++++++++++++++++++++++++++++++

- N=1
	- un el Alemania nació más " pasar adecuadas un armada que evadirse la
	- acomodadas residenciales Montpelhièr artesanos com , hidrato el ), al hasta territorio ensayos A obispo la batalla cercana Editorial arqueólogo de 969 cuevas y liga tercero DJ tarde como y algunas , cortarle una , fueron , art Se The conocer de . , . Balcanes método diciembre , de estaba implicó el río y , escribir país de Cholula las de que
	- inversibles como más antiguo . . y cero En católicos ocupa Goscinny . , ( destacados , importante un eran región de de Este aristócratas y de lo las 1979 medidas su ( - en
	- un
	- bodega de sus una son presumiblemente del medida contaba ellos comunista por una tercer con , New de de Esquemáticamente observables líquido longitudes ". una se sus : situó entre está 9 Alfonso llana Pablo España tuvo enfermedad La
- N=2
	- El propio derrame cerebral y objetos , la iniciación de las diferencias de la provincia , Viena , algunas de una futura , la iglesia de 1921 - 12 hasta 114 kbit / 3 . C .).
	- El fiasco .
	- Ocupa una personalidad y beben los votos y Oxford creó la mayoría de los sitios tradicionales .
	- Ellos realizaron las primeras fuerzas de Dawkins es una situación más grande la infancia transcurrió en Pekín cuando Paz .
	- En las que le produjo la dictadura militar y la encíclica , los teoremas menores , mezclaban elementos invertibles es también conocida también la decisión tuvo un bebe de Arnhem y fue constituido el último causante de Spengler " from : La historia egipcia encontramos ante Boca .
- N=3
	- Entre los colegios católicos italianos en el cual el estado de equilibrio .
	- Pet Shop Boys se remonta hasta Teispes en sus actos .
	- Más de la minería , agricultura ).
	- En Sri Lanka ( antigua Ceilán ), en el que de contrario usara todos los pares ordenados ( x , y se consigue más por sus propiedades ligeras y , además de imponérsele el pago de dos números enteros prescindiendo de la escuela asiática ( de 1820 .
	- Como reacción a la mitad del siglo XI a raíz del sínodo convocado por el poeta clásico de la balanza que medía el oro en grandes cantidades de nicotina .
- N=4
	- Años más tarde la fama no dañada A su regreso a Francia en octavos de final , donde vencieron por 3 - 0 en Anfield Road , aunque en Estados Unidos .
	- Algunos populares santuarios y mausoleos en la provincia .
	- El ludismo surgió como una gran mezquita y una catedral .
	- Las líneas pintadas en la carretera PR 22 , ocupando una superficie de 319 , 1 km² en la que ambas eran cogidas de manera diferente , o el anillo de enteros : suma , resta , y multiplicación .
	- En este país tomó parte en tres guerras a escala mundial ( Primera Guerra Mundial .

Ejercicio 4
-----------
Se hace a AddOneNGram subclase de NGram para heredar los métodos con la misma implementación. El método V() se implementó en NGram, pues es necesario también para InterpolatedNGram y BackOffNGram. Se implementó un método set_vocab_size, que calcula y guarda el tamaño del alfabeto en función de las oraciones. El único método que se reimplementa en AddOneNGram es cond_prob, para realizar el smoothing. Los demás son idénticos a los de NGram.

Ejercicio 5
-----------
Se agregaron cuatro nuevos métodos a la clase NGram: log_probability, cross_entropy, perplexity, logp_entropy_perplexity. Todos ellos toman la lista de oraciones de test. logp_entropy_perplexity devuelve una tupla con las tres métricas para el caso en que se necesiten las tres (por ejemplo, en el script eval.py), sin necesidad de recalcular valores. De esta forma, queda muy simple el script de evaluación, y además estos métodos resultarán útiles si se necesita estimar parámetros de un modelo en base a un conjunto held out.

Ejercicio 6
-----------
La clase InterpolatedNGram también hereda de NGram, sobreescribiendo el __init__ y cond_prob. A diferencia de NGram, InterpolatedNGram calcula los counts para todos los niveles de N-grama menores o iguales que n. Si el parámetro gamma no está dado, se estima mediante un barrido como el valor que minimiza la perplexity sobre el conjunto held-out (tomando una oración en cada intervalo de 10 de sents). Se definió un método auxiliar para calcular los lambdas en cond_prob: _lambdas_from_prev_tokens. De esta forma, cond_prob devuelve simplemente el producto punto entre el resultado de este método y las probabilidades maximum-likelihood de los niveles de k-gramas, k<=n.

Ejercicio 7
-----------
La clase BackOffNGram también hereda de NGram, sobreescribiendo el __init__ y cond_prob. En __init__ se guardan dos diccionarios más:

- card_a guarda el cardinal del conjunto A para cada secuencia de tokens, para obtener este valor eficientemente en el cálculo de alpha y denom. 
- sum_c guarda el siguiente valor para cada secuencia de tokens x1..xi: sum(count(x2..xix) for x in A(x1..xi)). Estos valores son útiles para calcular eficientemente denom.

Se implementa la función A, pero ineficientemente (recorriendo todo el diccionario counts), ya que esta función no se utiliza en el cálculo de cond_prob.

El cálculo de alpha se hace mediante la fórmula simplificada en función de beta, el cardinal de A y el count de los tokens.

El cálculo de denom se hace mediante un pequeño desarrollo de la fórmula, para utilizar el valor precalculado de sum_c y que no sea necesario recorrer todo el conjunto A.

La cond_prob utiliza los métodos alpha y denom para el caso recursivo.


Valores de perplexity
---------------------------

+---------------+----------+----------+----------+----------+
| Modelo        | N=1      | N=2      | N=3      | N=4      |
+===============+==========+==========+==========+==========+
| AddOne        | 1851.47  | 8355.15  | 44026.4  | 65411.8  |
+---------------+----------+----------+----------+----------+
| Interpolated  | 1846.33  | 873.89   | 874.14   | 887.39   |
+---------------+----------+----------+----------+----------+
| BackOff       | 1846.33  | 639.12   | 599.55   | 610.34   |
+---------------+----------+----------+----------+----------+


Instalación
-----------

1. Se necesita el siguiente software:

   - Git
   - Pip
   - Python 3.4 o posterior
   - TkInter
   - Virtualenv

   En un sistema basado en Debian (como Ubuntu), se puede hacer::

    sudo apt-get install git python-pip python3.4 python3-tk virtualenv

2. Crear y activar un nuevo
   `virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_.
   Recomiendo usar `virtualenvwrapper
   <http://virtualenvwrapper.readthedocs.org/en/latest/install.html#basic-installation>`_.
   Se puede instalar así::

    sudo pip install virtualenvwrapper

   Y luego agregando la siguiente línea al final del archivo ``.bashrc``::

    [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]] && source "/usr/local/bin/virtualenvwrapper.sh"

   Para crear y activar nuestro virtualenv::

    mkvirtualenv --system-site-packages --python=/usr/bin/python3.4 pln-2015

3. Bajar el código::

    git clone https://github.com/PLN-FaMAF/PLN-2015.git

4. Instalarlo::

    cd pln-2015
    pip install -r requirements.txt


Ejecución
---------

1. Activar el entorno virtual con::

    workon pln-2015

2. Correr el script que uno quiera. Por ejemplo::

    python languagemodeling/scripts/train.py -h


Testing
-------

Correr nose::

    nosetests


Chequear Estilo de Código
-------------------------

Correr flake8 sobre el paquete o módulo que se desea chequear. Por ejemplo::

    flake8 languagemodeling


TODO:

Agregar oraciones generadas con modelos de n-gramas (1,2,3,4)
Reportar resultados de perplexity para Interpolated, AddOne, BackOff
