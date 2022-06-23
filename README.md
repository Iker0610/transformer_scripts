# Finetuning para Tareas NER Empleando Transformers

> En este proyecto se encuentran los _ficheros_ y _scripts_ necesarios para hacer **finetuning** de **transformers** para tareas **NER** empleando la librería **HuggingFace**.


## Estructura del proyecto

    src
    ├── dataset_loader                                                  # Carpeta para guardar ficheros de carga de datasets
    │   └── conll_dataset_loader.py                                     # Fichero de ejemplo para cargar en la clase dataset de HuggingFace un dataset de tipo CONLL

    ├── model_params                                                    # Carpeta con ejemplos de ficheros JSON con los parámetros para el script de entrenamiento
    │   ├── eriberta_params.json                                        # Fichero JSON con todos los parámetros configurados (puede que haya más de los que se necesiten, pero no falta ninguno que sea necesario)
    │   └── plantilla_eriberta_params.json                              # Fichero JSON que servirá de plantilla, contiene palabras clave que habrá que sustituir. Emplear junto con el script para hacer varios experimentos
    
    ├── slurm_launchers                                                 # Carpeta con scripts de ejemplo para lanzar experimentos en el sistema de colas
    │   ├── run_transformer_finetuning-multiple_experiments.slurm       # Ejemplo para lanzar varios experimentos, útil para barrido de parámetro y comprobación de la varianza del modelo
    │   ├── run_transformer_finetuning-single-experiment.slurm          # Ejemplo para lanzar un único experimento
    │   └── EJEMPLO_OUTPUT-slurm-384.out                                # Ejemplo de output generado por una ejecución del proceso de finetuning
    
    └── scripts                                                         # Carpeta con los scripts
        ├── train_transformer_pytorch.py                                # Script principal, sirve para finetunear y/o predecir/evaluar. Admite sliding-window para superar el límite de 512 tokens, y genera un CONLL con los resultados
        ├── seqeval_allMetrics.py                                       # Versión editada del script de HuggingFace de las métricas seqeval para que devuelva el micro avg, macro avg y weighed avg.
        └── utils                                                       # Carpeta con scripts de utilidad
            ├── get_dataset_labels.py                                   # Script para obtener una lista con los tags/labels/clases únicos que haya presentes en el dataset 
            ├── experiment_cleaner.py                                   # Script encargado de limpiar una carpeta de experimentos, permite limpiar solo los checkpoints y mantener el mejor modelo o eliminar todo (menos los reusltados)
            └── get_run_data.py                                         # Dado una carpeta donde se han ejecutado varios experimentos permite agruparlos todos en un CSV listo para subirlo a Tableau (requiere ajustes)

------------------------------------------------------

## Como hacer el Finetuning y lanzar los ficheros

> **IMPORTANTE** -
> En esta sección del `README` se detallan pasos para organizar un proyecto en una estructura predeterminada que en general funciona para cualquier proyecto.
> Según la tarea es posible que la estructura sea distinta, los scripts y ficheros de configuración se deban guardar en otro sitio, hacer más copias y guardarlas en subcarpetas, etc.
>> **Por tanto**, la estructura de esta guía es un ejemplo, y se supone que se tiene experiencia suficiente como para alterar la estructura según las necesidades de la tarea a realizar.
>
>> Las ediciones que hay que hacer a los ficheros **NO SON TAN FLEXIBLES** a no ser que se sepa muy bien que se está haciendo y porqué
> [ al fin y al cabo no dejan de ser scripts y por tanto son muy flexibles si uno lo desea ].

### Paso 1 - Generar el proyecto:

- #### Paso 1.1: Clonar el repo
  Para comenzar copia el proyecto en tu carpeta y cámbiale el nombre a la carpeta `src` por uno que represente el experimento. **Ejemplo**: `/home/iker/experiments/src -> /home/user/experiments/transformer_evaluation`.
    ```shell
    cd /home/user/experiment
    git clone https://github.com/Iker0610/transformer_scripts.git
    mv src my_experimento_con_transformers
    ```
 
  &nbsp;
- #### Paso 1.2: Añadir una carpeta para el dataset
  Genera una carpeta `dataset` o `datasets` si vas a hacer el mismo experimento con varios datasets. En este caso genera dentro de `datasets` una carpeta por cada dataset.
    ```shell
    cd my_experimento_con_transformers
    mkdir -p datasets/dataset_1 # El -p sirve para que genere también las carpetas padre si no existen
    
    # O si solo vais a trabajar con un dataset
    cd my_experimento_con_transformers
    mkdir dataset
    ```
  
  &nbsp;
- #### Paso 1.3: Mover el dataset a la carpeta
  Copia tus archivos **CONLL** en la carpeta correspondiente. Se recomienda que sean 3 archivos: `train.conll`, `dev.conll`, `test.conll`.
    ```shell
    # Multiples datasets:
    cp /mi/carpeta/con/los/dataset/*.conll /home/user/experiment/datasets/dataset_1/
    
    # Un dataset
    cp /mi/carpeta/con/los/dataset/*.conll /home/user/experiment/dataset/
    ```
    - Si los tuvieras separados en múltiples archivos (ejem: `train/fich1.conll`, `train/fich2.conll` ...) tienes dos opciones:
        - *[Version fácil]* Junta todos los ficheros en un único CONLL, separalos por una cabecera tipo | Fichero nombre_fichero - |
        - *[Versión difícil (no demasiado)]* Edita el script principal para que acepte una lista de ficheros y después el fichero JSON para darle una lista de ficheros por cada subset.

  &nbsp;
  > **IMPORTANTE** - Por sencillez se pide mover los dataset, pero puede ser que esto no sea posible (porque son grandes o para evitar duplicados), en ese caso omitir los pasos **1.2** y **1.3**
  >> Este es uno de los casos a los que se hace referencia al principio de la sección.

------------------

### Paso 2 - Preparar el fichero de carga de los dataset

- #### Paso 2.1 - Generar el script para cargar el dataset
  Para ello emplearemos el archivo `conll_dataset_loader.py`.

    - Si habéis copiado los dataset en la carpeta correspondiente:
      ```shell
      # Estando en src
      cp /dataset_loader/conll_dataset_loader.py ./datsets/dataset_1/MY_DATASET_loader.py
      ```

    - Si no pues cambiarle el nombre o hacer una copia en esa misma carpeta:
      ```shell
      # Cambiarle el nombre
      mv /dataset_loader/conll_dataset_loader.py /dataset_loader/MY_DATASET_loader.py
    
      # O hacer copia
      cp /dataset_loader/conll_dataset_loader.py /dataset_loader/MY_DATASET_loader.py
      ```
  
  &nbsp;
  > **NOTA** - Por facilidad al fichero `MY_DATASET_loader.py` generado se le llamará `conll_dataset_loader.py`.
  
  &nbsp;
  
- #### Paso 2.2 - Adaptar el fichero de carga del dataset

  Estos son los parámetros que hay que ajustar siempre en vuestra copia de `conll_dataset_loader.py`:

  |     Líneas      | Descripción del cambio a realizar                                             | Ejemplo - Notas                                                                     |
  |:---------------:|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  |        7        | Aquí hay que indicar qué elemento representa el inicio de un nuevo documento. | En la sección donde se indica el tag del token se pone un '-' que no es un tag real |
  |       10        | Aquí se indica cual es el caracter que separa las columnas.                   | Normalmente es el espacio, aunque puede que en algún caso alguien use `\t`          |
  |    13-16-18     | Ajustar estos valores a los nombre de vuestro dataset.                        | -                                                                                   |
  | [ Opcional ] 23 | Lo mismo que en el punto anterior.                                            | -                                                                                   |
  |       32        | Indicar **TODAS** las clases (tags) que vayan a estar presentes.              | Usando el script `get_dataset_labels.py` se pueden obtener automáticamente.         |

  El script actual se basa en el siguiente tipo de fichero:
  ```
  # Columnas: token line-offset ner_tags
  File SP03103.conll -
  Hoy 0-3 O
  he 4-6 O
  visitado 7-15 O
  Bilbao 16-21 B-CIUDAD
  ...
  
  File SP03104.conll -
  ```
  En cada uno de estos conll se esperan 3 columnas en el orden indicado. Por cada fichero (en caso de que haya varios) se espera tener una cabecera donde el `ner_tag` deberá ser igual que el indicado en `_TAG_INI_FICH` _[línea 7]_.

  La función `predict_and_save_to_conll` del script `train_transformer_pytorch.py` se basa en este dataset, por tanto, en caso de que hayan más o menos columnas hay 2 opciones:
    - [Solución rápida pero sucia] Omitir las columnas que no estén indicadas. Y en caso de faltar en el conll alguna columna indicada (la única sería `line_offset`, si te falta otra tienes un problema) asignarle un valor arbitrario (como `'null-null'`) a mano en el script de carga.
    - [Solución un poco más larga pero correcta] Editar los ficheros `conll_dataset_loader.py` y `train_transformer_pytorch.py` para que tengan en cuenta las nuevas columnas:
      - En `conll_dataset_loader.py`:

        |      Lineas      | Descripción del cambio a realizar                                                                                                 |
        |:----------------:|:----------------------------------------------------------------------------------------------------------------------------------|
        |      26-28       | Editar este diccionario para añadir/eliminar/editar las columnas. Este diccionario representa la estructura de **UNA** instancia. |
        | 77-81 y 105-108  | Añadir / resetear variables incializadas con el tipo de dato apropiado (el que hayáis indicado en el las líneas 26-28.            |
        | 98-101 y 121-124 | Añadir a los diccionarios las variables que hayáis creado.                                                                        |
        |     115-117      | Añadir a las variables que sean lista el valor de la línea del CONLL.                                                             |

      - En `train_transformer_pytorch.py`:

        | Líneas | Descripción del cambio a realizar                                                                                                                                                                                                                                                                                                                                                                                                                     |
        |:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        |  521   | En el `zip` de este for debéis añadir / cambiar / eliminar los campos *(a partir de `prediction_dataset['overflow_to_sample_index_mapping'],`)* para que cuadren con los que habéis puesto, estas se guardarán en `*token_data`. <br/> **NO** Elimineis las secciones `id`, `overflow_to_sample_mapping`, `overflow_to_sample_index_mapping` y `true_predictions`. La columna ner_tags no es necesaria en este for.                                   |
        |  534   | Ajustar el formato de la línea que indica el inicio de documento (no os olvidéis del `\n` al final).                                                                                                                                                                                                                                                                                                                                                  |
        |  542   | Ajustar las variables de este for para que haya tantas como el número de columnas que hayáis guardado en `*token_data` en la línea 521. <br/> *NO* toqueis lo que hay después del `in list...`.                                                                                                                                                                                                                                                       |
        |  543   | Ajustar el formato de la línea del conll para que tenga todas las columnas que debería en el orden que debería (no os olvidéis del `\n` al final).                                                                                                                                                                                                                                                                                                    |
      
      &nbsp;
      > **MUY IMPORTANTE** - **Siempre** tendréis que tener unas secciones: `id`, `tokens` y `ner-tags`, _con esos mismos nombres_. [ `id` es autoincremental ].
      >> **RECORDATORIO** - Cuando editéis los ficheros los offset del resto de pasos se desajustará, así que siempre comprobad la sección a editar en GitHub o en una versión sin editar.

      > **DETALLES** - El script genera una instancia por cada archivo que localiza con el tag `_TAG_INI_FICH`. Es posible que sea necesario una instancia por cada párrafo, para lo que habría que editar el script de carga y seguramente la función que genera el conll en `train_transformer_pytorch.py`.
------------------------

### Paso 3 - Preparar el fichero de parámetros

- #### Paso 3.1 - Generar el fichero con los parámetros

  Para este paso hay que elegir primero qué se va a hacer:
    - Si solo vas a lanzar un experimento entonces haz una copia de `eriberta_params.json` y llámala de una forma que sepas a qué se refiere.
  ```shell
  cp model_params/eriberta_params.json model_params/MY_TRANSFORMER_MY_EXPERIMENT_params.json
  ```
    - Si vas a hacer varios experimentos, ya sea con los mismos parámetros (**distinta random seed**) o barrido de parámetros entonces necesitas una plantilla, para ello copia `plantilla_eriberta_params.json` y cámbiale el nombre.
  ```shell
  cp model_params/plantilla_eriberta_params.json model_params/plantilla_MY_TRANSFORMER_MY_EXPERIMENT_params.json
  ```
  &nbsp;
  > **NOTA** - Este es uno de los puntos donde seguramente debas generar varios y guardarlos en sitios distintos o no.

  &nbsp;

- #### Paso 3.2 - Actualizar los ficheros de parámetros
  En este paso hay que actualizar el fichero generado para ajustar los valores de los parámetros. En la siguiente tabla se dan más detalles:

| Parámetro                                                    |                                   Valor Recomendado                                    | Descripción                                                                                                                                                                                                                                                                                                                                                                                                 |
|:-------------------------------------------------------------|:--------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `do_train` / `do_eval` / `do_predict`                        |                               `true` / `true` / `false`                                | Definen si al ejecutar ese script hay que hacer finetuning, ejecutar la predicción sobre dev y predecir y evaluar test.<br/> Como norma general al hacer finetuning usar los valores recomendados. <br/> Si ya tienes un modelo y solo quieres predecir entonces activa `do_eval` y `do_test` según lo que necesites.<br/> `do_eval` y `do_test` se comportan exactamente igual pero con subsets distintos. |
| `train_file` / `dev_file` / `test_file`                      |                             *Path absoluto a los ficheros*                             | Indican el path al fichero que contiene cada uno de los subsets. `train_file` y `dev_file` son necesarios para el finetuning.<br/> Se recomienda poner los 3 y ajustar qué hacer con `do_train` / `do_eval` / `do_predict`.                                                                                                                                                                                 |
| `dataset_loading_script`                                     |                               *Path absoluto al fichero*                               | Path al script que se empleará para cargar el dataset a la librería `datasets` de *Huggingface*. Este es el fichero que se hha generado partiendo de `conll_dataset_loader.py`.                                                                                                                                                                                                                             |
| `dataset_cache_dir`                                          |                              *Path absoluto a la carpeta*                              | Path a una carpeta donde se guardarán los archivos caché relativos al dataset.                                                                                                                                                                                                                                                                                                                              |
| `overwrite_cache`                                            |                                        `false`                                         | Si sobreescribir los archivos cache del dataset. En principio debería ser `false` con excepción de si se edita el dataset y se emplea el mismo loading script o si se nota algo raro.                                                                                                                                                                                                                       |
|                                                              |                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                             |
| `model_path`                                                 |                              *Path absoluto a la carpeta*                              | Carpeta donde se encuentran los ficheros necesarios para el modelo: `my_model.bin`, `config.json` y todos los ficheros relativos a su tokenizer.                                                                                                                                                                                                                                                            |
| `output_dir`                                                 |                              *Path absoluto a la carpeta*                              | Carpeta donde se generarán todos los ficheros de salida y checkpoints. (No es necesario que la carpeta exista, el script crea todas las carpetas necesarias)                                                                                                                                                                                                                                                |
| `overwrite_output_dir`                                       |                                        `false`                                         | Por seguridad es mejor que sea siempre `false`, de esta forma no se puede destruir accidentalmente una carpeta de un experimento. Si es `true` borra todo lo que haya en la carpeta si es que hay algo.                                                                                                                                                                                                     |
|                                                              |                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                             |
| `pad_to_max_length`                                          |                                         `true`                                         | En principio siempre, SIEMPRE, debe ser `true`. No se me ocurre ningún caso donde se acabe entrenando un transformer en una CPU.                                                                                                                                                                                                                                                                            |
| `use_sliding_window`                                         |                                         `true`                                         | Si es `true` entonces siempre clasificará todos los tokens de las instancias empleando una sliding-window. En principio no interesa que sea `false` a excepción de reproducir otro experimento que lo haya hecho así.                                                                                                                                                                                       |
| `stride_size`                                                |                                      `50` - `150`                                      | Nº de tokens que se solapan al aplicar sliding-window. Es recomendable que no sea muy pequeño para darle contexto pero tampoco excesivamente grande.                                                                                                                                                                                                                                                        |
| `return_entity_level_metrics`                                |                                         `true`                                         | Te devuelve la métrica por cada clase, en principio es recomendable que sea `true`.                                                                                                                                                                                                                                                                                                                         |
|                                                              |                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                             |
| `learning_rate`                                              |                                         `5e-5`                                         | El por defecto es `5e-5` y funciona bastante bien, por supuesto hay que optimizarlo para la tarea (truco rápido `*2` y `/2`).<br/> Últimamente se está probando por learning rates muy pequeños y más epochs.                                                                                                                                                                                               |
| `num_train_epochs`                                           |                              `~10-20` _(según la tarea)_                               | Número de epochs, ajustar según los resultados y la tarea. Al ser finetuning con unos pocos debería ser suficiente.                                                                                                                                                                                                                                                                                         |
| `per_device_train_batch_size` / `per_device_eval_batch_size` |                           `32` *(los que entren en la GPU)*                            | El óptimo es tener un `batch size` de **mínimo** `32` aunque no siempre entran en la GPU, en ese caso usa un múltiplo de `2` y ajustar `gradient_accumulation_steps` hasta que cuadre.                                                                                                                                                                                                                      |
| `gradient_accumulation_steps`                                |    *Los necesarios para que `train_batch_size * gradient_accumulation_steps = 32`*     | ^                                                                                                                                                                                                                                                                                                                                                                                                           |
|                                                              |                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                             |
| `evaluation_strategy`                                        |                                        `epoch`                                         | Cada cuanto ejecutar una evaluación sobre el subset `dev`/`eval`. En principio cada epoch está bien.<br/> Si se cambia a `steps` hay que añadir y ajustar `eval_steps`.                                                                                                                                                                                                                                     |
| `save_strategy`                                              |                                        `epoch`                                         | Cada cuanto guardar un checkpoint. Si se cambia a steps hay que añadir y ajustar `save_steps`.                                                                                                                                                                                                                                                                                                              |
| `metric_for_best_model`                                      |                                       `micro_f1`                                       | Métrica empleada para elegir el mejor modelo.                                                                                                                                                                                                                                                                                                                                                               |
| `save_total_limit`                                           |                                          `1`                                           | Máximos modelos de checkpoint a tener. Depende el espacio y la tarea se recomienda entre 1 y 3 *(es obligatorio al menos uno para elegir el mejor modelo)*.                                                                                                                                                                                                                                                 |
| `load_best_model_at_end`                                     |                                         `true`                                         | Si es `true` (que es siempre lo recomendado) se queda con el mejor modelo que haya salido. <br/>Requiere que `evaluation_strategy` y `save_strategy` estén configurados igual y lo mismo para `eval_steps` y `save_steps` si hemos decidido que se usen `steps`.                                                                                                                                            |
|                                                              |                                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                             |
| `seed`                                                       | *Integer random y que no hayas usado en otro experimento con el mismo modelo/dataset.* | Seed para los números random, se recomienda obtener uno del sistema (ejem: `shuf -i 0-5500 -n 1`) y que quede registrado y sea distinto de cualquiera usado para pruebas similares.                                                                                                                                                                                                                         |
| `fp16`                                                       |                                         `true`                                         | Un modo especial que usa floats con menos precisión en el training lo que acelera el entrenamiento ***casi** sin afectar a los resultados*. En el finetuning no causa tanto impacto como a la hora de entrenar un modelo desde 0.                                                                                                                                                                           |

&nbsp;

> **IMPORTANTE** - Si vas a hacer varios experimentos variando los parámetros se recomienda generar una plantilla [ indicado en el paso 3.1 ] y ponerles a los parámetros variables un valor arbitrario aunque no sea correcto para poder sustituirlos dinámicamente.
>> *Ejemplo:* `"seed": VAR_SEED,` , `VAR_SEED` no es un valor válido pero resulta sencillo buscarlo con `sed` y sustituirlo dinámicamente.
> 
>> *TODO:* En un futuro quizá mejore la optimización de parámetros emnpleando un `.py` con **Ray-Tune** y no haga falta un `.json` plantilla.


------------------------

### Paso 4 - Lanzar el script en los servidores

- #### Paso 4.1 - Ajustar los scripts de lanzamiento
  Los scripts de lanzamiento son:
  - `run_transformer_finetuning-single-experiment.slurm`: Para ejecutar un único experimento, este script espera que le des un `MY_TRANSFORMER_params.json` completamente configurado [ ver sección 3.2 ]. Se encarga de borrar los **checkpoint**, pero no el modelo final.
  - `run_transformer_finetuning-multiple_experiments.slurm`: Para ejecutar varios experimentos del mismo tipo (con objetivo de evaluar la varianza) y con distintos learning rates; espera que se le de una `plantilla_MY-MODELO_params.json` para generar un `MY-MODELO_params.json`. Se encarga de generar una carpeta única por cada experimento y borrar todos los checkpoints y modelos dejando solo los resultados.
  
  &nbsp;
  > **NOTA** - Ambos scripts se encargan de generar un copia del `params.json` en el directorio de salida (`output_dir`), de esta forma siempre quedará claro qué parámetros se emplearon para ese experimento (incluida la `seed`).
  >> **IMPORTANTE** - Simpre tiene que quedar registro de qué parámetros se ha empleado en que ejecución.

&nbsp;
  
  En estos scrips hay varias variables que hay que ajustar a los paths correctos. 
  - Cambiar el valor de la `línea 3` por un nombre que identifique vuestro proceso en la cola.
  - Aseguraos de que `OUTPUT_PATH` es igual (o una carpeta padre según el caso) que el parámetro `output_dir` en `MY_TRANSFORMER_params.json`.
  - Aseguraos de que `SRC_PATH` cuadra con la carpeta donde tengáis los scripts.
  - Revisar el resto de variables que hay y ser ordenados y cuidadosos sobre todo si tenéis varios modelos y/o varios datasets. 

- #### Paso 4.2 - Lanzar el script

  > **RECOMENDACIÓN** - Antes de lanzar el script en una máquina que acepte el sistema de colas es recomendable mirar si alguna máquina está libre. Para ello usar:
  > ```shell
  > squeue      # Para comprobar cuanta gente hay en la cola
  > nvidia-smi  # Para comprobar la disponibilidad de las GPU de esa máquina
  > ```
  
  Una vez que el script está listo:

  - Moveros a la carpeta donde se encuentra:
    ```shell
    cd ./slurm_launchers
    ```
  - Lanzar el script empleando y **LISTO**:
    ```shell
    sbatch run_transformer_finetuning-multiple_experiments.slurm # O como sea que hayáis llamado al script
    ```
  
  &nbsp;
- #### Paso 4.3 - Monitorizar la ejecución
  Aunque el script se haya lanzado aún pueden pasar muchas cosas, el script no funciona, no entra en cola, algún parámetro no es correcto, algo no se ejecuta como debería...
  
  - Para comprobar el estado de nuestro script en la cola (y su `id`):
    ```shell
    squeue
    ```
  - Para comprobar si nuestro proceso ha entrado en la GPU y cuanta memoria usa:
    ```shell
    nvidia-smi
    ```
  - Para cancelar un proceso en la cola [ porque ha salido mal ]:
    ```shell
    scancel id_del_proceso # Mucho ojo no le vayáis a cancelar el proceso a otro !!!!!!!!
    ```
  - Para ver el output por consola de nuestro proceso slurm genera en la misma carpeta desde la que se ha lanzado el `sbatch` un fichero con nombre `slurm-{id_del_proceso}` que se va actualizando.

  &nbsp;
  > **IMPORTANTE** - Ir comprobando los ficheros de output para ver que va todo según lo planeado, sobre todo al principio. 
  > Si se generan modelos / checkpoints y estos deberían borrarse aseguraos que en efecto se borran correctamente al acabar el experimento. Si no lo hacen cancelar, limpiar y mirar porqué no funcionan.