"""[summary]

"""
resource_folder = "resource files"
user_def_stop_words = ['a', 'ante', 'con','por','si', 'la','en' , 'el',\
        'que', 'del', 'los', 'las', 'para', 'como', 'los','fue', 'al', 'su',\
        'se','un', 'bajo', 'lo', 'e', 'me', 'le', 'sus', 'su', 'hasta',\
        'sin', 'o', 'u','ha', 'han', 'y', 'este','esta', 'derechos', 'ley',\
        'asi', 'cual', 'articulo', 'cuando', 'y']

# MODEL DATA RESOURCE PARAMETERS
name_csv = "Funcionarios-Table_1.csv"
gen_csv = 'nombres_gen.csv' 

# WORD LIST RESOURCE PARAMETERS
words_url = 'https://github.com/javierarce/palabras/blob/master/listado-general.txt' 
words_list_local = "spanish_words.txt"
words_list_internal = "spanish_words.txt"

# MODEL TRAINING PAREMETERS
alphas=[0.1, 1, 5, 10] # params to test MultinomialNB classifiers