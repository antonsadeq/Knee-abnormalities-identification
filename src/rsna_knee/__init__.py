"""RSNA Knee Abnormality Detection (Kaggle 2026)."""

from .config import LABELS, LATERALITY_PAIRS, STUDY_ID
from .labeller import label_corpus, label_report, training_targets
from .metric import macro_auc, per_class_auc, score_frames
from .reports import Clause, detect_language, fold, segment, unwrap
from .verify import coverage, disagreement, gold_agreement, polarity

__all__ = [
    "LABELS",
    "LATERALITY_PAIRS",
    "STUDY_ID",
    "Clause",
    "coverage",
    "detect_language",
    "disagreement",
    "fold",
    "gold_agreement",
    "label_corpus",
    "label_report",
    "macro_auc",
    "per_class_auc",
    "polarity",
    "score_frames",
    "segment",
    "unwrap",
]
