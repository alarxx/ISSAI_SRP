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

## Nvidia Drivers on Linux Debian 12 "Bookworm"
https://wiki.debian.org/NvidiaGraphicsDrivers

**Check devices**

```sh
lspci -nn | egrep -i "3d|display|vga"
```

**Prerequisites**

```sh
apt install linux-headers-amd64 build-essential
```

**Driver**

Add to `/etc/apt/sources.list` file `contrib` and `non-free`:
```txt
# Debian Bookworm
deb http://deb.debian.org/debian/ bookworm main contrib non-free non-free-firmware
```
(Warning: Don't add Sid, it will brake your system)

"proprietary" flavor:
```sh
apt update
apt install nvidia-driver firmware-misc-nonfree
```

**CUDA**

```sh
apt install nvidia-cuda-dev nvidia-cuda-toolkit
```

**Check**

```shell
# Driver
nvidia-smi
# CUDA
nvcc --version
```

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
pip install nvitop
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
source init.sh
```

---

## Distributed Data Parallel (DDP)

В основном, в DDP, синхронизация идет в constructor (grad hooks), forward (BatchNorm) и backward pass (ring all-reduce). Дополнительно, мы можем добавить свои точки синхронизации с помощью `barrier()`.
Если один процесс загружен больше или там слабее видеокарта, другие процессы могут простаивать в ожидании.

## Licence

Tensor-library is licensed under the terms of [MPL-2.0](https://mozilla.org/MPL/2.0/), which is simple and straightforward to use, allowing this project to be combined and distributed with any proprietary software, even with static linking. If you modify, only the originally covered files must remain under the same MPL-2.0.

License notice:
```
SPDX-License-Identifier: MPL-2.0
--------------------------------
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.

This file is part of the ISSAI Summer Research Project.

Description: <Description>

Provided “as is”, without warranty of any kind.

Copyright © 2025 Alar Akilbekov. All rights reserved.

Third party copyrights are property of their respective owners.
```

---

Если вы собираетесь модифицировать MPL покрытые файлы (IANAL):
- Ваш fork проект = MPL Covered files + Not Covered files (1.7).
- MPL-2.0 is file-based weak copyleft действующий только на Covered файлы, то есть добавленные вами файлы и исполняемые файлы, полученные из объединенения с вашими, могут быть под любой лицензией (3.3).
- (но под copyleft могут подпадать и новые файлы в которых copy-paste-нули код из Covered) (1.7).
- Покрытыми лицензией (Covered) считаются файлы с license notice (e.g. .cpp, .hpp) и любые исполняемые виды этих файлов (e.g. .exe, .a, .so) (1.4).
- You may not remove license notices (3.4), как и в MIT, Apache, BSD (кроме 0BSD) etc.
- При распространении любой Covered файл должен быть доступен, но разрешено личное использование или только внутри организации (3.2).
- Если указан Exhibit B, то производную запрещается лицензировать с GPL.
- Contributor дает лицензию на любое использование конкретной интеллектуальной собственности (patent), которую он реализует в проекте (но не trademarks).

Эти разъяснения условий не меняют и не вносят новые юридические требования к MPL.

---

## Contact

Alar Akilbekov - alar.akilbekov@gmail.com
