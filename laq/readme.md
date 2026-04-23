python3 -m venv lapa_env
source lapa_env/bin/activate
source .env/bin/activate
source .venv/bin/activate

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


env -u LD_LIBRARY_PATH bash train_ssv2.sh

bash train_ssv2.sh