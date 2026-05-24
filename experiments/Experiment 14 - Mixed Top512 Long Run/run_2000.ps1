$ErrorActionPreference = "Stop"

rtk python -u "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" `
    --steps 2000 `
    --variants dense_tied_vocab,mixed_top512 `
    --device cuda

exit $LASTEXITCODE
