import librosa
import numpy as np
from pathlib import Path
import shutil
import cv2
import os
from src.utils import normalize_melspec
from fastprogress import progress_bar

# Parameters
TARGET_SR = 32000
melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}
pcen_parameters = {
    "gain": 0.98,
    "bias": 2,
    "power": 0.5,
    "time_constant": 0.4,
    "eps": 0.000001
}
PERIOD = 30
CHUNK = PERIOD * TARGET_SR

###


def transform_all_images(dirpath: str, sound_file: str, csv_file: str):
    """Create a folder with .png for the training"""
    csv_file = open(dirpath + csv_file, "r", encoding='utf-8')
    i = 0
    csv_file.readline()

    # Reset the temp folder
    shutil.rmtree('train/temp/train', ignore_errors=True)
    os.makedirs('train/temp/train')
    shutil.rmtree('train/temp/val', ignore_errors=True)
    os.makedirs('train/temp/val')

    for audio_line in progress_bar(csv_file.readlines()):
        L = audio_line.split(",")
        id_audio = L[-2]
        id_species = L[1]
        # Create a folder for each species
        os.makedirs('train/temp/train/'+id_species, exist_ok=True)
        os.makedirs('train/temp/val/'+id_species, exist_ok=True)

        image = np.swapaxes(clip_to_image(
            dirpath+sound_file+id_audio, all_chunks=False), 0, 2)

        # 70% of the audio are used for the training phase and 30% for the validation phase
        if i % 50 > 15:
            cv2.imwrite('train/temp/train/'+id_species +
                        '/'+id_audio+'.png', image)

        else:
            cv2.imwrite('train/temp/val/'+id_species +
                        '/'+id_audio+'.png', image)
        i += 1


def preproc(y):
    """return the preprocessing of a clip 'y' """
    y_batch = y.astype(np.float32)

    if len(y_batch) > 0:  # Normalization
        max_vol = np.abs(y_batch).max()
        if max_vol > 0:
            y_batch = np.asfortranarray(y_batch * 1 / max_vol)

    # Zero paddling to have an input of constant size
    y_pad = np.zeros(PERIOD * TARGET_SR, dtype=np.float32)
    y_pad[:len(y_batch)] = y_batch

    # spectrograms
    melspec = librosa.feature.melspectrogram(y=y_pad,
                                             sr=TARGET_SR,
                                             **melspectrogram_parameters)
    pcen = librosa.pcen(melspec, sr=TARGET_SR, **pcen_parameters)
    clean_mel = librosa.power_to_db(melspec ** 1.5)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    # Normalization
    norm_melspec = normalize_melspec(melspec)
    norm_pcen = normalize_melspec(pcen)
    norm_clean_mel = normalize_melspec(clean_mel)
    # Concatenate, we have a color picture
    image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width * 224 / height), 224))
    image = np.moveaxis(image, 2, 0)
    image = (image).astype(np.float32)

    return image


def clip_to_image(clip_path: str, all_chunks=True):
    """return the clip almost ready to apply the model. If all_chunks=False, only the first chunk is returned"""
    # load the audio file
    if Path(clip_path+".mp3").exists():

        clip, _ = librosa.load(clip_path+".mp3",
                               sr=TARGET_SR,
                               mono=True,
                               res_type="kaiser_fast")
    elif Path(clip_path+".wav").exists():
        clip, _ = librosa.load(clip_path + ".wav",
                               sr=TARGET_SR,
                               mono=True,
                               res_type="kaiser_fast")
    try:
        clip
    except UnboundLocalError:
        raise FileExistsError(
            f"{clip_path}.mp3 or .wav doesn't exist, only .wav & .mp3 are allowed. Aswell, it might be an audio from the .csv that is not in the audio folder. Easy fix : delete the corresponding line in the csv")

    y = clip.astype(np.float32)

    if not all_chunks:
        image = preproc(y[:CHUNK])
        array = np.asarray(image)
        return (array)

    nb_chunk = (len(y)-1)//CHUNK+1
    images = []
    for k in range(nb_chunk):
        image = preproc(y[k*CHUNK:(k+1)*CHUNK])
        images.append(image)
    array = np.asarray(images)
    return (array)
