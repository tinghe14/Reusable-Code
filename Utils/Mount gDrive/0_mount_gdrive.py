from google.colab import drive
drive.mount('/content/drive/')
import os

#copy the custom module from google drive to colab temporary drive
!cp /content/drive/MyDrive/Ting/data_process.py /content/data_process.py

if not os.path.exists('/content/input'):
  os.makedirs('/content/input', exist_ok=True)
!cp /content/drive/MyDrive/Ting/input/devel.tsv /content/input/devel.tsv
!cp /content/drive/MyDrive/Ting/input/test.tsv /content/input/test.tsv
!cp /content/drive/MyDrive/Ting/input/train.tsv /content/input/train.tsv
