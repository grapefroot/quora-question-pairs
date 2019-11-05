# Quora-question-pairs

Solution for quora [Quora question pairs challenge](https://www.kaggle.com/c/quora-question-pairs/) using pretrained BERT models

Steps to reproduce:
1. Initialize git submodules `git submodule update --init --recursive`
2. Make sure you have installed packages from [requirements](requirements.txt)
3. Download quora-pairs-dataset.zip and unzip it to `./data` (create if missing)
4. Download checkpoint weights for models from google drive 
[model1](https://drive.google.com/open?id=1uh3VaWkUCNj9S3-1Uv63CClrEo6WP7cu) [model2](https://drive.google.com/open?id=1wHXh7Gn1GbPpsva_0tHm2ybpoJ7AHUfJ) 
and put them into `./models` (create if missing)

Additionally, [script](setup.sh) was created to help you automate this,
but in case it doesn't work for you just make above steps manually

Now, everything is ready.
You may replicate the submission by running in Python

```python
from utils import replicate
replicate(YOUR_PATH)
```

which will create submission csv with specified path
