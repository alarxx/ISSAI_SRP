#! /bin/sh

# chmod +x init.sh

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

touch .env
# We can use GPUs:{2,3,4,5}
# Append to file: `echo "CUDA_VISIBLE_DEVICES=3,4" >> .env`
# Overwrite the file:
echo "CUDA_VISIBLE_DEVICES=3,4" > .env
