prepare:
	python creation_datasets/create_csv.py France
	python creation_datasets/download_files.py France.csv
	rm -rf train/data_training/birds.csv
	mv creation_datasets/fichiers_csv/France.csv train/data_training/birds.csv
	rm -rf train/data_training/audio_files/
	mv creation_datasets/audio_files_France train/data_training/audio_files/
	python preload_training_data.py

train:
	python train.py -m all