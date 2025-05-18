import argparse
import json
import torch
from transformers import AutoTokenizer
from src.model import MultiTaskModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Инференс мультитаск-модели: токены, POS-теги и CLS-эмбеддинг"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Путь к папке с сохранённой моделью (save_pretrained)"
    )
    parser.add_argument(
        "--label_map", type=str, required=True,
        help="JSON-файл с маппингом POS-тегов в id (label->int)"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Путь к текстовому файлу (.txt) с проверяемым текстом"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Устройство: cuda или cpu (по умолчанию auto-detect)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Загрузка текста из файла
    with open(args.input_file, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("Файл пустой, нечего анализировать.")
        return

    # Настройка устройства
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Загрузка маппинга id->label
    with open(args.label_map, encoding="utf-8") as f:
        label_map = json.load(f)
    id2pos = {v: k for k, v in label_map.items()}

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = MultiTaskModel.from_pretrained(args.checkpoint).to(device)
    model.eval()

    # Токенизация и инференс
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        pos_logits, embedding = model(**inputs)

    # Разбор результатов
    pred_ids = pos_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    token_ids = inputs["input_ids"].squeeze(0).cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Собираем итоговые токены и метки, пропуская специальные
    result_tokens, result_tags = [], []
    for tok, pid in zip(tokens, pred_ids):
        if tok in tokenizer.all_special_tokens:
            continue
        result_tokens.append(tok)
        result_tags.append(id2pos.get(pid, "X"))

    # CLS-эмбеддинг
    cls_embedding = embedding.squeeze(0).cpu().tolist()

    # Вывод
    print("TOKENS:", result_tokens)
    print("POS TAGS:", result_tags)
    print("EMBEDDING vector length:", len(cls_embedding))

if __name__ == "__main__":
    main()