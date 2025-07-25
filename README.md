# ISSAI_SRP
> ISSAI Summer Research Project

## Install Miniconda

https://www.anaconda.com/docs/getting-started/miniconda/install#linux

## Connect to GitHub

https://github.com/alarxx/ISSAI_SRP/blob/how_to_connect/README.md

## References

- https://github.com/vladimiralbrekhtccr/ISSAI_SRP_2025
- https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- https://huggingface.co/learn/llm-course

---

## Getting Started

In this project I use python venv, 
because conda have some bugs, it doesn't isolate the project as expected.

Install APT packages:
```sh
apt install python3 python3-pip python3-venv python3-tk
# apt install python3.8
```

Create and use environment:
```sh
python3 -m venv .venv
# -m - module-name, finds sys.path and runs corresponding .py file
source .venv/bin/activate
```

Install python libraries:
```sh
pip install torch torchvision torchaudio
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install python-dotenv
```

To recreate environment:
```sh
pip freeze > requirements.txt
```

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or just execute:
```sh
./init.sh
```

---

## Distributed Data Parallel