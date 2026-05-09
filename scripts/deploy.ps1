$ErrorActionPreference = "Stop"

python -m scripts.deploy @args
exit $LASTEXITCODE
