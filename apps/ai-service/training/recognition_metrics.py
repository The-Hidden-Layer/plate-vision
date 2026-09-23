"""Paired-crop recognition metrics shared by standalone and full-pipeline evaluation."""

from PIL import Image

from app.lpr.alphabet import LETTERS, tokenize


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = next_row
    return row[-1]


def validate_recognizer_data(recognizer, fingerprint):
    if recognizer.dataset_fingerprint != fingerprint:
        raise ValueError("recognizer checkpoint and evaluation dataset fingerprints differ")
    if recognizer.smoke_only:
        raise ValueError("accuracy reports require a full trained recognizer, not smoke weights")


def evaluate_recognizer(recognizer, rows, root, batch_size, *, predictions=None):
    rows = [row for row in rows if row["dataset"] == "LPR"]
    if not rows or batch_size < 1:
        raise ValueError("recognition evaluation requires paired crops and a positive batch size")
    by_letter = {
        letter: {"count": 0, "correct": 0, "character_errors": 0, "characters": 0}
        for letter in LETTERS
    }
    exact = edits = symbols = unreadable = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        crops = []
        for row in batch:
            if len(row["labels"]) != 1:
                raise ValueError("recognition evaluation requires exactly one label per crop")
            with Image.open(root / row["image"]) as source:
                crops.append(source.convert("RGB"))
        reads = recognizer.read_batch(
            crops, job_id="evaluation", frame_index=offset, slots=list(range(len(batch)))
        )
        for row, crop, read in zip(batch, crops, reads, strict=True):
            text = row["labels"][0]
            expected = tokenize(text)
            predicted = tokenize(read.text) if read.text else []
            correct = text == read.text
            errors = edit_distance(expected, predicted)
            if predictions is not None:
                predictions.append(
                    {
                        "image": row["image"],
                        "expected": text,
                        "text": read.text,
                        "confidence": read.confidence,
                        "correct": correct,
                        "character_errors": errors,
                        "width": crop.width,
                        "height": crop.height,
                    }
                )
            exact += correct
            edits += errors
            symbols += len(expected)
            unreadable += not read.text
            letter = by_letter[expected[2]]
            letter["count"] += 1
            letter["correct"] += correct
            letter["character_errors"] += errors
            letter["characters"] += len(expected)
    for values in by_letter.values():
        values["exact_accuracy"] = values["correct"] / values["count"] if values["count"] else None
        values["character_error_rate"] = (
            values["character_errors"] / values["characters"] if values["characters"] else None
        )
    return {
        "exact_accuracy": exact / len(rows),
        "correct": exact,
        "character_error_rate": edits / symbols,
        "character_errors": edits,
        "characters": symbols,
        "unreadable_count": unreadable,
        "count": len(rows),
        "per_letter": by_letter,
    }
