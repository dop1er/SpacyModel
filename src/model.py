import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig


class MultiTaskConfig(PretrainedConfig):
    """
    Конфигурация для мультитаск-модели.
    Хранит название энкодера и число меток POS.
    """
    model_type = "multitask"

    def __init__(
        self,
        encoder_name_or_path: str = "DeepPavlov/rubert-base-cased",
        num_pos_labels: int = 17,
        proj_size: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.num_pos_labels = num_pos_labels
        self.proj_size = proj_size


class MultiTaskModel(PreTrainedModel):
    """
    Мультитаск модель: энкодер + две "головы":
      - pos_head: классификация POS на каждый токен
      - proj: проекция CLS-эмбеддинга для векторизации
    """
    config_class = MultiTaskConfig
    base_model_prefix = "multitask_model"

    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        # Загружаем конфигурацию энкодера отдельно
        encoder_cfg = AutoConfig.from_pretrained(config.encoder_name_or_path)
        # Загружаем сам энкодер с игнорированием лишних весов
        self.encoder = AutoModel.from_pretrained(
            config.encoder_name_or_path,
            config=encoder_cfg,
            ignore_mismatched_sizes=True
        )
        hidden_size = encoder_cfg.hidden_size
        # Голова для POS-разметки
        self.pos_head = nn.Linear(hidden_size, config.num_pos_labels)
        # Проекция CLS для эмбеддингов
        proj_dim = config.proj_size or hidden_size
        self.proj = nn.Linear(hidden_size, proj_dim)
        # Инициализация весов мультитаск-голов
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        # Получаем выходы энкодера
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_output = sequence_output[:, 0, :]       # [batch, hidden]

        # Логиты для POS (по токенам)
        pos_logits = self.pos_head(sequence_output)  # [batch, seq_len, num_pos_labels]
        # Векторные эмбеддинги из CLS
        proj_output = self.proj(cls_output)          # [batch, proj_dim]

        return pos_logits, proj_output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Переопределяем from_pretrained, чтобы корректно загружать MultiTaskConfig и энкодер.
        """
        config = MultiTaskConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    def save_pretrained(self, save_directory):
        """
        Сохраняем config, weights и токенизатор.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
