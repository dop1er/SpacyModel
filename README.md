# SpacyModel

Запуск обучения:
python -m src.train --labeled_data data/raw/ru_syntagrus-ud-2.16 --pos_label_map configs/pos_label_map.json --morph_label_map configs/morph_label_map.json --encoder_name DeepPavlov/rubert-base-cased --output_dir checkpoints/multitask_model --epochs 3 --batch_size 8 --lr 2e-5 --device cuda

Запуск модели (после обучения):
В файле data/input.txt ввести текст, далее строка
python infer.py --checkpoint checkpoints/multitask_model/epoch-3 --label_map configs/pos_label_map.json --input_file data/input.txt --device cuda
