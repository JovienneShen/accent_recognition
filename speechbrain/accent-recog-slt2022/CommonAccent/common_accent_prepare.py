"""
Data preparation of CommonAccent dataset for Accent classification (English).
(Recipe that uses CV version 11.0)
Download: https://commonvoice.mozilla.org/en/datasets

Author
------
 * Juan Pablo Zuluaga 2023
"""

import csv
import logging
import os
import argparse
import re
import warnings
import unicodedata

import pandas as pd
import torchaudio
from speechbrain.utils.data_utils import get_all_files
from tqdm.contrib import tzip

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

""" You need to first run : download_data_hf.py, to create the TSV files, and
    Then you need to select the accents you want to use in the training.

    For instance, you can do in the terminal:
    - cat /out/folder/tsv/files/* | cut -d$'\t' -f5 | sort | uniq -c | sort -n

    That command produces the count number of labeled samples with that accent. 
    Select accents with at least 100 samples.
"""

RAW_PATH = 'data/cv_11_raw/'

_ACCENTS_EN = [
    7e4, # max 20000 samples per accent
    # "Austrian", # 104
    # "East African Khoja", # 107
    # "Dutch", # 108
    # "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)", # 282
    # "Welsh English", # 623
    # "Malaysian English", # 1004
    # "Liverpool English,Lancashire English,England English", # 2571
    # "Singaporean English", # 2792
    # "Hong Kong English", # 2951
    # "Filipino", # 4030
    # "Southern African (South Africa, Zimbabwe, Namibia)", # 4270
    # "New Zealand English", # 4960
    # "Irish English", # 6339
    # "Northern Irish", # 6862
    # "Scottish English", # 10817
    # "Australian English", # 33335
    # "German English,Non native speaker", # 41258
    # "Canadian English", # 45640
    "England English", # 75772
    # "India and South Asia (India, Pakistan, Sri Lanka)", # 79043
    "United States English", # 249284
]
_ACCENTS_FR = [
    1e4, # max 10000 samples per accent
    # "Français d’Algérie", # 319 
    # "Français d’Allemagne", # 355 
    # "Français du Bénin", # 823 
    # "Français de La Réunion", # 884 
    # "Français des États-Unis", # 898 
    "Français de Suisse", # 3608 
    "Français de Belgique", # 6509 
    "Français du Canada", # 8073 
    "Français de France", # 342921
]
_ACCENTS_DE = [
    1e4, # max 10000 samples per accent
    "Italienisch Deutsch", # 947 
    "Schweizerdeutsch", # 9891 
    "Österreichisches Deutsch", # 16066 
    "Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch", # 50843 
    "Deutschland Deutsch", # 252709
]
_ACCENTS_IT = [
    1e4, # max 10000 samples per accent
    "Emiliano", # 151
    "Meridionale", # 193
    "Veneto", # 1508
    "Tendente al siculo, ma non marcato", # 2175
    "Basilicata,trentino", # 2297
]
_ACCENTS_ES = [
    1e4, # max 10000 samples per accent
    # "España: Islas Canarias", # 1326
    "Chileno: Chile, Cuyo", # 4285
    # "América central", # 6031
    "Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México", # 8329
    # "España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)", # 8683
    "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay", # 11162
    "Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina", # 12997
    "México", # 26136
    # "España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca)", # 30588
    "España: Sur peninsular (Andalucia, Extremadura, Murcia)", # 38251
]


def prepare_common_accent(
        data_folder, 
        save_folder, 
        accented_letters=False,
        language="en",        
        skip_prep=False,
    ):
    """
    Prepares the csv files for the CommonAccent dataset for Accent Classification.
    Download: https://commonvoice.mozilla.org/en/datasets

    Arguments
    ---------
    data_folder : str
        Path to the folder where the CommonAccent dataset for Accent Classification is stored.
        This path should include the multi: /datasets/CommonAccent
    save_folder : str
        The directory where to store the csv files.
    max_duration : int, optional
        Max duration (in seconds) of training uterances.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.        
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonAccent.common_accent_prepare import prepare_common_accent
    >>> data_folder = '/datasets/CommonAccent'
    >>> save_folder = 'exp/CommonAccent_exp'
    >>> prepare_common_accent(\
            data_folder,\
            save_folder,\
            skip_prep=False\
        )
    """

    if skip_prep:
        return

    # Setting the save folder
    os.makedirs(save_folder, exist_ok=True)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, "train.csv")
    save_csv_dev = os.path.join(save_folder, "dev.csv")
    save_csv_test = os.path.join(save_folder, "test.csv")

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):
        csv_exists = " already exists, skipping data preparation!"
        msg = save_csv_train + csv_exists
        logger.info(msg)
        msg = save_csv_dev + csv_exists
        logger.info(msg)
        msg = save_csv_test + csv_exists
        logger.info(msg)

        return

    # Additional checks to make sure the data folder contains Common Accent
    check_common_accent_folder(data_folder, language=language)

    # Audio files extensions
    extension = [".mp3"]

    # Create the signal list of train, dev, and test sets.
    data_split = create_sets(data_folder, extension, language=language)

    # Creating csv files for training, dev and test data
    create_csv(wav_list=data_split["train"], csv_file=save_csv_train)
    create_csv(wav_list=data_split["dev"], csv_file=save_csv_dev)
    create_csv(wav_list=data_split["test"], csv_file=save_csv_test)


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the CommonAccent data preparation for Accent Classification has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    )

    return skip

import ipdb
def create_sets(data_folder, extension, language="en"):
    """
    Creates lists for train, dev and test sets with data from the data_folder

    Arguments
    ---------
    data_folder : str
        Path of the CommonAccent dataset.
    extension: list of file extentions
        List of strings with file extentions that correspond to the audio files
        in the CommonAccent dataset

    Returns
    -------
    dictionary containing train, dev, and test splits.
    """
    # get the ACCENT dictionary from accent_configuration
    if language == "en":
        ACCENTS = _ACCENTS_EN
    elif language == "it":
        ACCENTS = _ACCENTS_IT
    elif language == "de":
        ACCENTS = _ACCENTS_DE
    elif language == "fr":
        ACCENTS = _ACCENTS_FR
    elif language == "es":
        ACCENTS = _ACCENTS_ES
    
    # get the max_samples_per_accent from the list:
    max_samples_per_accent = ACCENTS[0]

    # accent counter to balance the datasets:
    accent_counter = { acc_id: 0 for acc_id in ACCENTS}

    # Datasets initialization
    datasets = {"train", "validation", "test"}

    # Get the list of accents from the dataset folder
    msg = f"Loading the data of train/validation/test sets!"
    logger.info(msg)

    accent_wav_list = []

    # Fill the train, dev and test datasets with audio filenames
    for dataset in datasets:
        curr_csv_file = os.path.join(data_folder, language, dataset + ".tsv")
        with open(curr_csv_file, "r", encoding='utf-8') as file:
            csvreader = csv.reader(file, delimiter='\t')
            for row in csvreader:
                if not row:
                    continue
                accent = row[4]  # accent information is in this field

                # if accent is part of the accents (top file dict), then, we add it:
                if accent in ACCENTS:
                    
                    # check if we have reached the max_samples per accent:
                    if accent_counter[accent] > max_samples_per_accent:
                        continue
                    utt_id = row[1]
                    wav_path = RAW_PATH + row[2]

                    # get transcript and clean it! 
                    transcript = clean_transcript(row[7], language=language)
                    # short transcript, remove:
                    if len(transcript.split()) < 1: continue

                    # also we clean the label, which will be used during training
                    clean_accent = clean_transcript(accent, language=language)

                    # Peeking at the signal (to retrieve duration in seconds)
                    if os.path.isfile(wav_path):
                        info = torchaudio.info(wav_path)
                        audio_duration = info.num_frames / info.sample_rate
                    else:
                        msg = "\tError loading: %s" % (str(len(wav_path)))
                        logger.info(msg)
                        continue
                    # append to list
                    accent_wav_list.append([utt_id, "$data_root/" + wav_path, transcript, audio_duration, clean_accent])

                    # update the accent counter
                    accent_counter[accent] += 1

    print(accent_counter)

    # Split the data in train/dev/test balanced:
    df = pd.DataFrame(
        accent_wav_list, columns=["utt_id", "path", "transcript", "duration", "accent"]
    )

    df_train, df_dev, df_test = [], [], []

    # We need to create the train/dev/test sets, with equal samples for dev and test sets
    all_accents = df.accent.unique()

    # for loop to go over each accent and get the values
    for accent in all_accents:
        condition = df["accent"] == accent

        # subset with only the given 'accent'
        df_with_accent = df[condition]
        df_size = int(df_with_accent.accent.count())

        # if there are less than 500 samples, we put 20% for dev and test sets, 60% for train
        n_samples = int(df_size * 0.1) if df_size > 500 else int(df_size * 0.2)

        # get and append the first 100 values for dev/test sets, for train, we use the rest
        df_dev.append(df_with_accent.iloc[0:n_samples])
        df_test.append(df_with_accent.iloc[n_samples : n_samples * 2])
        df_train.append(df_with_accent.iloc[n_samples * 2 :])


    # create the object with the pandas DataFrames to output:
    accent_wav_list = {}
    accent_wav_list["train"] = pd.concat(df_train)
    accent_wav_list["dev"] = pd.concat(df_dev)
    accent_wav_list["test"] = pd.concat(df_test)

    msg = "Data successfully loaded!"
    logger.info(msg)

    return accent_wav_list


def create_csv(wav_list, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    csv_file : str
        The path of the output json file
    """

    # Adding some Prints
    msg = f"Creating csv lists in {csv_file} ..."
    logger.info(msg)

    csv_lines = []

    # Start processing lines
    total_duration = 0.0

    # Starting index
    idx = 0
    for sample in wav_list.iterrows():
        
        # get some data from the file (CommonVoice is MP3)
        utt_id = sample[1][0]
        wav_path = sample[1][1]
        wav_format = wav_path.split(os.path.sep)[-1].split(".")[-1]
        transcript = sample[1][2]
        accent = sample[1][4]
        audio_duration = sample[1][3]

        # Create a row with whole utterences
        csv_line = [
            idx,  # ID
            utt_id,  # Utterance ID
            wav_path,  # File name
            wav_format,  # File format
            transcript, # transcript
            str("%.3f" % audio_duration),  # Duration (sec)
            accent,  # Accent
        ]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

        # Increment index
        idx += 1

        # update total duration
        total_duration += audio_duration


    # CSV column titles
    csv_header = ["ID", "utt_id", "wav", "wav_format", "text", "duration", "accent"]

    # Add titles to the list at indexx 0
    csv_lines.insert(0, csv_header)

    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = f"{csv_file} sucessfully created!"
    logger.info(msg)
    msg = f"Number of samples: {len(wav_list)}."
    logger.info(msg)
    msg = f"Total duration: {round(total_duration / 3600, 2)} hours."
    logger.info(msg)


def check_common_accent_folder(data_folder, language='en'):
    """
    Check if the data folder actually contains the CommonAccent dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain at least two accents.
    """

    # Checking if at least two accents are present in the data
    files = set(os.listdir(os.path.join(data_folder, language)))

    if "train.tsv" not in files:
        err_msg = f"{data_folder}/{language} must have at least 'train.tsv' file in it."
        raise FileNotFoundError(err_msg)


def main():
    # read input from CLI, you need to run it from the command lind
    parser = argparse.ArgumentParser()

    # reporting vars
    parser.add_argument(
        "--language",
        type=str,
        default='en',
        help="Language to load",
    )
    parser.add_argument(
        "cv_folder",
        help="path of the input folder, where CV dataset is stored (files in TSV format)",
    )
    parser.add_argument(
        "output_folder",
        help="path of the output folder to store the csv files for each split",
    )
    # parse the arguments and run the preparation    
    args = parser.parse_args()
    prepare_common_accent(args.cv_folder, args.output_folder, language=args.language)

def clean_transcript(words, language='en', accented_letters=False):
    """ function to clean the input transcript 
        input:
        words: transcript
        language: language of the sample, default=en
        accented_letters: whether to remove accented_letters
    """
    def unicode_normalisation(text):
        try:
            text = unicode(text, "utf-8")
        except NameError:  # unicode is a default on python 3
            pass
        return str(text)
    
    # Unicode Normalization
    words = unicode_normalisation(words)

    # !! Language specific cleaning !!
    # Important: feel free to specify the text normalization
    # corresponding to your alphabet.

    def strip_accents(text):
        text = (
            unicodedata.normalize("NFD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        return str(text)

    if language in ["en", "fr", "it", "rw"]:
        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()

    if language == "de":
        # this replacement helps preserve the case of ß
        # (and helps retain solitary occurrences of SS)
        # since python's upper() converts ß to SS.
        words = words.replace("ß", "0000ß0000")
        words = re.sub("[^’'A-Za-z0-9öÖäÄüÜß]+", " ", words).upper()
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        words = words.replace(
            "0000SS0000", "ß"
        )  # replace 0000SS0000 back to ß as its initial presence in the corpus

    if language == "fr":
        # Replace J'y D'hui etc by J_ D_hui
        words = words.replace("'", " ")
        words = words.replace("’", " ")

    elif language == "ar":
        HAMZA = "\u0621"
        ALEF_MADDA = "\u0622"
        ALEF_HAMZA_ABOVE = "\u0623"
        letters = (
            "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
            + HAMZA
            + ALEF_MADDA
            + ALEF_HAMZA_ABOVE
        )
        words = re.sub("[^" + letters + " ]+", "", words).upper()
    elif language == "ga-IE":
        # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
        def pfxuc(a):
            return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

        def galc(w):
            return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

        words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
        words = " ".join(map(galc, words.split(" ")))

    # regex with predifined symbols to ignore/remove,
    chars_to_ignore_regex2 = '[\{\[\]\<\>\/\,\?\.\!\u00AC\;\:"\\%\\\]|[0-9]'

    words = re.sub(chars_to_ignore_regex2, "", words)

    # Remove accents if specified
    if not accented_letters:
        words = strip_accents(words)
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        words = words.replace("\u2013", "-")
        words = words.replace("\u2014", "-")
        words = words.replace("\u2018", "'")
        words = words.replace("\u201C", "")
        words = words.replace("\u201D", "")
        words = words.replace("ñ", "n")
        words = words.replace(" - ", " ")
        words = words.replace("-", "")
        words = words.replace("'", " ")

    # Remove multiple spaces
    words = re.sub(" +", " ", words)

    # Remove spaces at the beginning and the end of the sentence
    words = words.lstrip().rstrip().upper()
    
    return words


# Recipe begins! (when called from CLI)
if __name__ == "__main__":
    main()
