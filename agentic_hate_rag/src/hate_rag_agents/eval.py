"""Auto-eval engine: tracks per-row metrics, computes summary statistics, and writes eval reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any


@dataclass
class EvalTracker:
    """Accumulates per-row ground-truth vs prediction pairs and computes running metrics."""

    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    misclassified: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        gold_label: int,
        predicted_label: int,
        confidence: float,
        text: str = "",
        normalized_text: str = "",
        explanation: str = "",
        row_index: Any = "",
    ) -> None:
        self.y_true.append(gold_label)
        self.y_pred.append(predicted_label)
        self.confidences.append(confidence)
        if gold_label != predicted_label:
            self.misclassified.append(
                {
                    "tracker_index": len(self.y_true) - 1,
                    "row_index": row_index,
                    "text": text,
                    "normalized_text": normalized_text,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

    @property
    def total(self) -> int:
        return len(self.y_true)

    def pop_recent_misclassified(self, since: int) -> list[dict[str, Any]]:
        """Return misclassified items added since the tracker had *since* total items."""
        return [item for item in self.misclassified if isinstance(item.get("tracker_index"), int) and item["tracker_index"] >= since]

    def get_batch_misclassified(self, batch_start: int, batch_end: int) -> list[dict[str, Any]]:
        """Return misclassified items whose indices fall within the batch range."""
        return [
            item
            for item in self.misclassified
            if isinstance(item.get("row_index"), Integral) and batch_start <= item["row_index"] < batch_end
        ]

    def metrics(self) -> dict[str, Any]:
        """Compute accuracy, precision, recall, F1, and confusion matrix from accumulated results."""
        if not self.y_true:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "total": 0}

        tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == 1 and p == 0)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total": total,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "misclassified_count": len(self.misclassified),
            "mean_confidence": round(sum(self.confidences) / len(self.confidences), 4) if self.confidences else 0.0,
        }


def generate_eval_summary(tracker: EvalTracker, output_path: Path) -> dict[str, Any]:
    """Write a JSON evaluation summary and return the metrics dict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = tracker.metrics()
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def print_eval_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable evaluation summary to the console."""
    cm = summary.get("confusion_matrix", {})
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total rows evaluated:  {summary.get('total', 0)}")
    print(f"  Accuracy:              {summary.get('accuracy', 0.0):.2%}")
    print(f"  Precision:             {summary.get('precision', 0.0):.2%}")
    print(f"  Recall:                {summary.get('recall', 0.0):.2%}")
    print(f"  F1 Score:              {summary.get('f1', 0.0):.2%}")
    print(f"  Mean Confidence:       {summary.get('mean_confidence', 0.0):.2%}")
    print(f"  Misclassified:         {summary.get('misclassified_count', 0)}")
    print()
    print("  Confusion Matrix:")
    print(f"                     Predicted=0    Predicted=1")
    print(f"    Actual=0           TN={cm.get('tn', 0):<8}   FP={cm.get('fp', 0)}")
    print(f"    Actual=1           FN={cm.get('fn', 0):<8}   TP={cm.get('tp', 0)}")
    print("=" * 60 + "\n", flush=True)
