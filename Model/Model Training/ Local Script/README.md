# README

Code finished by my own on June 05, 2023. 

Originally, this scrpts is for ner task using bert model. 

# Structure
- 📂 __Gene\_NER\_Clean\_Script\_wocluster__
   - 📂 __input__
     - 📄 some files
     - 📂 __clean__
       - 📄 some files
   - 📄 [config.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/config.py)
   - 📄 [data\_process.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/data_process.py)
   - 📄 [main.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/main.py)
   - 📄 [model.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/model.py)
   - 📄 [predict.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/predict.py)
   - 📄 [test.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/test.py)
   - 📄 [train.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/train.py)
   - 📄 [utils.py](https://github.com/tinghe14/Reusable-Code/blob/5c5840f6193a6bc7046cc25af9053d582660eabe/Model/Model%20Training/%20Local%20Script/Gene_NER_Clean_Script_wocluster/utils.py)
   - 📂 __output__
      - 📄 some plots and files

# Descriptions
- note: this code can run on gpu and cpu locally, but can't upload to cloud 
- config.py contains hyperparameter information and some varaibles releated to study setting
- after running main.py script, it will tokenize the words, train the bert model, save model checkpoints(weight and optimizer) based on best validation loss, save best and final result reports, save plot of validation accuracy among epochs for overall tags and major tags, and plot of training and validation loss among epochs
- test.py will load the last and best models and check their performance on the test dataset

# Reference
1. [陈华编程-Pytorch Bert_BiLSTM_CRF_NER 中文医疗命名实体识别项目](http://www.ichenhua.cn/edu/course/24): help me transfer from spaghetti code to script and I learn how to debug the script
2. [Saving and Loading the Best Model in PyTorch](https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/): learn how to implement early stopping based on validation loss, save model and optimizer checkpoint, save results(preformance and loss/accuracy plot) automatically. The tutorial wrote by this author recommend to read more
