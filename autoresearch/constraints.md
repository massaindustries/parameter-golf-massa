# Parameter Golf Challenge — Hard Constraints

These rules are immutable. Any experiment violating them is INVALID.

## Artifact Size
- Total artifact = code bytes (train_gpt.py UTF-8 encoded) + compressed model bytes (int8+zlib)
- Maximum: 16,000,000 bytes (decimal, NOT 16 MiB)
- No external downloads or network calls allowed during evaluation

## Training Budget (final submission)
- Maximum 10 minutes wall clock on 8×H100 SXM GPUs
- During the current autoresearch phase, comparison runs use the full 20,000-step protocol on the available local GPU.
- Do not use reduced 500-step or 2000-step runs as the basis for keep/discard decisions.

## Evaluation Budget (final submission)
- Maximum 10 minutes wall clock on 8×H100 SXM GPUs
- Evaluation must be fully self-contained and reproducible

## Metric
- `val_bpb` (bits per byte) on the fixed FineWeb validation set (first 50,000 documents)
- Lower is better
- Tokenizer-agnostic: BPB normalizes across different tokenizers
- Must be calculated after int8+zlib quantization roundtrip

## Test-Time Training Rules
- TTT is allowed ONLY on tokens already evaluated (backward-looking)
- You CANNOT pre-adapt on validation data before evaluating it
- You CANNOT access the training set during evaluation
- Any data used for TTT adaptation must already have been scored/graded

## Submission Requirements (for final PR)
- Beat existing SOTA by >= 0.005 nats
- Statistical significance p < 0.01 (typically 3 seeds required)
- Must include: README.md, submission.json, train.log, train_gpt.py
- Script must compile and run successfully

## Forbidden
- Training on the validation set
- External data or pre-trained weights
- Network calls during evaluation
- Brute-forcing seeds or sneaking in external compute
- Modifying the BPB calculation to unjustly improve score
