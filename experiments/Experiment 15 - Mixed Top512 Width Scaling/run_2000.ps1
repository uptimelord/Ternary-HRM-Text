$ErrorActionPreference = "Stop"

rtk python -u "experiments/Experiment 15 - Mixed Top512 Width Scaling/mixed_top512_width.py" `
    --steps 2000 `
    --hidden-sizes 192,256 `
    --device cuda `
    --append-md "experiments/Experiment 15 - Mixed Top512 Width Scaling/results_2000.md"

exit $LASTEXITCODE
