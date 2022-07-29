Drug Activity Prediction
This repository is the implementation of the Drug Activity prediction that predicts whether a given compound is active or not.

Installation
Requirements
- Python 3.7+
- DVC
- Flask

Setup
create env

```bash
 conda create -n drug_activation
```

activate env
```bash
conda activate drug_activation
```

created a req file
install the req
```bash
pip install -r requirements.txt
```

download the data

```bash
git init
```

```bash
dvc init
```

```bash
dvc add data_given/train.dat
```

```bash
git add .
```

```bash
git commit -m "first commit"
```

online updates for readme

```bash
git add . && git commit -m "update Readme.md"
```

```bash
git remote add origin https://github.com/Dhrupti02/drug_activity_prediction.git
git branch -M main
git push origin main
```

tox command -

```bash
tox 
```

for rebuilding -
```bash
tox -r
```

pytest command
```bash
pytest -v
```

setup commands -
```bash
pip install -e .
```