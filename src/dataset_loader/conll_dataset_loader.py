# coding=utf-8

import datasets

# TODO: AJUSTAR
# Variable que indica el inicio de un fichero en el dataset:
_TAG_INI_FICH = '-'

# Separador de las columnas
_SEP = ' '


class PharmaconerConll(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="pharmaconer_conll_loader",
            version=datasets.Version("1.0.0"),
            description="Dataset para la tarea 1 de Pharmaconer"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Dataset para Section Identification",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "line_offset": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            # TODO CAMBIAR POR LAS CABECERAS DE LAS SECCIONES
                            names=[
                                'O',
                                'B-NORMALIZABLES', 'I-NORMALIZABLES',
                                'B-NO_NORMALIZABLES', 'I-NO_NORMALIZABLES',
                                'B-PROTEINAS', 'I-PROTEINAS',
                                'B-UNCLEAR', 'I-UNCLEAR'
                            ]
                        )
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        # Se comprueba que se ha especificado el path a los ficheros:
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")

        # Se obtienen los path:
        data_files = dl_manager.download_and_extract(self.config.data_files)

        splits = []
        # Si solo se ha pasado un conjunto se carga como el split del train
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files}))

        # Si se ha pasado un diccionario, se guardan los distintos splits:
        else:
            for split_name, files in data_files.items():
                if isinstance(files, str):
                    files = [files]
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))

        return splits

    def _generate_examples(self, files):
        for filepath in files:
            # Abrimos el fichero, lo leemos y se hace splitt
            with open(filepath, encoding="utf-8") as f:
                file_lines = f.read().splitlines()

            # Creamos las variables donde se guardaran los datos:
            guid: int = 0
            fich_name: str = ''
            tokens: list[str] = []
            offset: list[str] = []
            ner_tags: list[str] = []

            # Iteramos sobre las líneas del fichero:
            for line in file_lines:

                # Si es una línea en blanco la saltamos
                if not line: continue

                # Dividimos las lineas por el correspondiente separador
                splits = line.split(_SEP)

                # Comprobamos si es el inicio de un nuevo documento:
                if splits[-1] == _TAG_INI_FICH:

                    # Si ya hay tokens (un fichero anterior) se mandan los datos
                    if tokens:
                        yield guid, {
                            "id": fich_name,
                            "tokens": tokens,
                            "line_offset": offset,
                            "ner_tags": ner_tags,
                        }

                        # Se resetean las variables para un nuevo fichero
                        guid += 1
                        tokens = []
                        offset = []
                        ner_tags = []

                    # Guardamos el nombre del nuevo fichero
                    fich_name = splits[1]

                # Si es una línea normal se guardan los datos
                else:
                    tokens.append(splits[0].strip())
                    offset.append(splits[2].strip())
                    ner_tags.append(splits[-1].strip())

            # El último fichero (que nunca va a ser seguido por otro):
            yield guid, {
                "id": fich_name,
                "tokens": tokens,
                "line_offset": offset,
                "ner_tags": ner_tags,
            }
