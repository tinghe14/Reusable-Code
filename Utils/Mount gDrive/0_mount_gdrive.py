from google.colab import drive
drive.mount('/content/drive/')
import os

os.makedirs('/content/input', exist_ok=True)
!cp /content/drive/MyDrive/Ting/input/*/content/input/devel.tsv
## cp:
# -r copy all files and subdirectories recursively
## double quotes in linux command
# single quote may not work as expected: only work when you want everything inside to be treated literally
# double quotes not works as expected when having * wildcard: we can use without double quote or use double quote as "example_path/"*
# when having space in path: must use double quote or escape character as example\ parent/example_path

!cp /content/drive/MyDrive/Ting/data_process.py /content/data_process.py


