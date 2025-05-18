import os
import glob
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Union

from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_unlabeled_texts(path: str, extensions: List[str] = None) -> Dataset:
    """
    Загружает неразмеченный корпус из файла или директории.
    Поддерживает форматы .txt (каждая строка — документ) и .jsonl (каждая строка — JSON с полем 'text').

    :param path: путь к файлу или директории с файлами.
    :param extensions: список расширений для поиска файлов, например ['txt', 'jsonl'].
    :return: Dataset с полем 'text'.
    """
    if extensions is None:
        extensions = ['txt', 'jsonl']
    texts = []
    files = []
    if os.path.isdir(path):
        for ext in extensions:
            files.extend(glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Path {path} not found.")

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append({'text': line})
        elif ext == '.jsonl':
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if 'text' in obj:
                            texts.append({'text': obj['text']})
                    except json.JSONDecodeError:
                        continue
        else:
            # игнорировать неизвестные форматы
            continue
    return Dataset.from_list(texts)


def group_texts(examples: Dict[str, List], block_size: int = 512) -> Dict[str, List[List[int]]]:
    """
    Сшивает входные тексты в длинные блоки фиксированной длины для обучения Masked LM.
    Используется после токенизации (map с batched=True).
    """
    concatenated = sum(examples['input_ids'], [])
    total_length = len(concatenated) // block_size * block_size
    chunks = [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]
    return {
        'input_ids': chunks,
        'attention_mask': [[1] * block_size for _ in chunks]
    }


def load_labeled_conllu(conllu_file: str) -> List[Dict[str, List[str]]]:
    """
    Загружает один файл CoNLL-U.
    Возвращает список примеров с ключами:
      - 'tokens': List[str]
      - 'pos_tags': List[str]
      - 'morph_feats': List[str]
    """
    examples, tokens, pos_tags, feats = [], [], [], []
    with open(conllu_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({'tokens': tokens, 'pos_tags': pos_tags, 'morph_feats': feats})
                    tokens, pos_tags, feats = [], [], []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if '-' in parts[0] or '.' in parts[0]:
                continue
            tokens.append(parts[1])
            pos_tags.append(parts[3])
            feats.append(parts[5] if parts[5] != '_' else '')
    if tokens:
        examples.append({'tokens': tokens, 'pos_tags': pos_tags, 'morph_feats': feats})
    return examples


def load_labeled_conllu_dir(directory: str) -> List[Dict[str, List[str]]]:
    """
    Загружает все файлы .conllu из директории (рекурсивно).
    Возвращает объединённый список примеров.
    """
    examples = []
    for file_path in glob.glob(os.path.join(directory, '**/*.conllu'), recursive=True):
        examples.extend(load_labeled_conllu(file_path))
    return examples


def load_labeled_opencorpora_xml(xml_file: str) -> List[Dict[str, List[str]]]:
    """
    Загружает файл OpenCorpora XML-формата. Каждый sentence -> tokens.
    Предполагается структура:
      <sentence>
        <token id="...">
          <text>слово</text>
          <tfr>
            <ana lex="..." gr="POS,..."/>
          </tfr>
        </token>
      </sentence>
    Возвращает примеры с 'tokens', 'pos_tags', 'morph_feats'.
    """
    examples = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for sent in root.findall('.//sentence'):
        tokens, pos_tags, feats = [], [], []
        for tok in sent.findall('.//token'):
            text_elem = tok.find('text')
            ana_elem = tok.find('.//ana')
            if text_elem is None or ana_elem is None:
                continue
            word = text_elem.text
            gr = ana_elem.get('gr', '')
            pos = gr.split(',')[0] if gr else ''
            tokens.append(word)
            pos_tags.append(pos)
            feats.append(gr)
        if tokens:
            examples.append({'tokens': tokens, 'pos_tags': pos_tags, 'morph_feats': feats})
    return examples


def tokenize_and_align_labels(
    examples: List[Dict[str, List[str]]],
    tokenizer: PreTrainedTokenizer,
    pos_label2id: Dict[str, int],
    morph_label2id: Dict[str, int],
    max_length: int = 512
) -> Dict[str, List[List[int]]]:
    """
    Токенизирует входные предложения и выравнивает метки POS и морфо-признаков по сабтокенам.
    Незначащие сабтокены получают метки -100.

    Возвращает словарь с полями:
      - input_ids, attention_mask
      - pos_labels
      - morph_labels
    """
    tokenized = tokenizer(
        [ex['tokens'] for ex in examples],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    all_pos, all_morph = [], []
    for i, ex in enumerate(examples):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        pos_ids, morph_ids = [], []
        for wid in word_ids:
            if wid is None:
                pos_ids.append(-100)
                morph_ids.append(-100)
            elif wid != prev:
                label_pos = ex['pos_tags'][wid]
                label_morph = ex['morph_feats'][wid]
                pos_ids.append(pos_label2id.get(label_pos, pos_label2id.get('X', -100)))
                morph_ids.append(morph_label2id.get(label_morph, morph_label2id.get('', -100)))
            else:
                pos_ids.append(-100)
                morph_ids.append(-100)
            prev = wid
        all_pos.append(pos_ids)
        all_morph.append(morph_ids)
    tokenized['pos_labels'] = all_pos
    tokenized['morph_labels'] = all_morph
    return tokenized


def load_labeled(path: str) -> List[Dict[str, List[str]]]:
    """
    Универсальная загрузка размеченных данных по расширению:
      - .conllu или директория с .conllu
      - .xml (OpenCorpora)
    """
    if os.path.isdir(path):
        return load_labeled_conllu_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.conllu':
        return load_labeled_conllu(path)
    elif ext in ['.xml', '.bz2', '.zip']:
        return load_labeled_opencorpora_xml(path)
    else:
        raise ValueError(f"Неподдерживаемый формат для размеченных данных: {path}")
