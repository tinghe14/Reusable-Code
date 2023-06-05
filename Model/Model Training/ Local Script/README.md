# README

My code finished on June 05, 2023. This scrpts originally is for ner task using bert model. 

# Structure
- 📂 __Gene\_NER\_Clean\_Script\_wocluster__
   - 📂 __input__
     - 📄 [devel.tsv](input/devel.tsv)
     - 📄 [merge.tsv](input/merge.tsv)
     - 📄 [test.tsv](input/test.tsv)
     - 📄 [train.tsv](input/train.tsv)
     - 📂 __clean__
       - 📄 [tag\_set.npy](input/clean/tag_set.npy)
       - 📄 [test\_sent.npy](input/clean/test_sent.npy)
       - 📄 [test\_tag.npy](input/clean/test_tag.npy)
       - 📄 [train\_sent.npy](input/clean/train_sent.npy)
       - 📄 [train\_tag.npy](input/clean/train_tag.npy)
   - 📄 [config.py](config.py)
   - 📄 [data\_process.py](data_process.py)
   - 📄 [main.py](main.py)
   - 📄 [model.py](model.py)
   - 📄 [predict.py](predict.py)
   - 📄 [test.py](test.py)
   - 📄 [train.py](train.py)
   - 📄 [utils.py](utils.py)
   - 📂 __output__

# Descriptions
- can run on gpu and cpu locally, but can't upload to cloud 
- after running main.py script, it will generate merge.tsv in input folder, all the npy files in clean folder and saved model checkpoints(weight and optimizer), best and final result report, plot of validation accuracy among epochs for overall tags and major tags, and plot of training and validation loss among epochs
