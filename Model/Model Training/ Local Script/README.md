# README

Code finished by my own on June 05, 2023. 

Originally, this scrpts is for ner task using bert model. 

# Structure
- 📂 __Gene\_NER\_Clean\_Script\_wocluster__
   - 📂 __input__
     - 📄 devel.tsv
     - 📄 merge.tsv
     - 📄 test.tsv
     - 📄 train.tsv
     - 📂 __clean__
       - 📄 tag\_set.npy
       - 📄 test\_sent.npy
       - 📄 test\_tag.npy
       - 📄 train\_sent.npy
       - 📄 train\_tag.npy
   - 📄 config.py
   - 📄 data\_process.py
   - 📄 main.py
   - 📄 model.py
   - 📄 predict.py
   - 📄 test.py
   - 📄 train.py
   - 📄 utils.py
   - 📂 __output__

# Descriptions
- can run on gpu and cpu locally, but can't upload to cloud 
- after running main.py script, it will generate merge.tsv in input folder, all the npy files in clean folder and saved model checkpoints(weight and optimizer), best and final result report, plot of validation accuracy among epochs for overall tags and major tags, and plot of training and validation loss among epochs

# Reference
1. [陈华编程-Pytorch Bert_BiLSTM_CRF_NER 中文医疗命名实体识别项目](http://www.ichenhua.cn/edu/course/24): help me transfer from spaghetti code to script and I learn how to debug the script
2. [Saving and Loading the Best Model in PyTorch](https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/): learn how to implement early stopping based on validation loss, save model and optimizer checkpoint, save results(preformance and loss/accuracy plot) automatically. The tutorial wrote by this author recommend to read more
