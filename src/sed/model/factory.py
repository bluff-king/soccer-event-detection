"""Model/tokenizer construction.

``model.name`` is any Hugging Face sequence-classification backbone (``xlm-roberta-base``,
``microsoft/mdeberta-v3-base``, ...) or ``tiny-local``: a randomly initialised 2-layer BERT with a
WordPiece tokenizer trained on the training texts. ``tiny-local`` needs no download, so the full
pipeline can be smoke-tested offline on CPU; its scores are meaningless.
"""

from __future__ import annotations

from pathlib import Path

from ..labels import ID2LABEL, LABEL2ID


def build_tiny_tokenizer(texts: list[str], vocab_size: int = 4000):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size,
                                        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tk.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


def build_model_and_tokenizer(cfg: dict, train_texts: list[str] | None = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig, BertForSequenceClassification

    name = cfg["model"]["name"]
    if name == "tiny-local":
        tok = build_tiny_tokenizer(train_texts or ["goal card penalty"], cfg["model"].get("tiny_vocab", 4000))
        h = cfg["model"].get("tiny_hidden", 64)
        conf = BertConfig(vocab_size=len(tok), hidden_size=h, num_hidden_layers=cfg["model"].get("tiny_layers", 2),
                          num_attention_heads=max(2, h // 64), intermediate_size=2 * h,
                          max_position_embeddings=max(512, cfg["model"]["max_length"] + 8),
                          num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
                          pad_token_id=tok.pad_token_id)
        return BertForSequenceClassification(conf), tok
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(LABEL2ID), id2label=ID2LABEL,
                                                               label2id=LABEL2ID)
    return model, tok


def load_checkpoint(path: str | Path, device: str = "cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device).eval()
    return model, tok
