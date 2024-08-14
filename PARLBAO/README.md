

1. Create conda environment
```
conda create -n PARL python=3.8
conda activate PARL
```

2. Install dependencies

```
paddlepaddle==2.2.0
parl==2.0.3
pip install setuptools==65.5.0 pip==21
gym==0.18.0
atari-py==0.2.6
rlschool==0.3.1
```
Attention, we may face problem when installing gym 0.18.0, please refer to the following link
https://stackoverflow.com/questions/77124879/pip-extras-require-must-be-a-dictionary-whose-values-are-strings-or-lists-of


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


