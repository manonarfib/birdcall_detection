from src.preproc import transform_all_images
if __name__ == '__main__':
    # Make the preprocessing and save .png in the train/temp/ folder
    transform_all_images('train/data_training/', 'audio_files/', 'birds.csv')
