# Reusable-Code
This repo contains my previous code which can be reuse later or great coding example which I found online.

markdown syntax shortcut:
- highlighter:
$`\textcolor{red}{\text{1}}`$ 
$`\textcolor{blue}{\text{2}}`$ 
$`\textcolor{green}{\text{3}}`$
- shortcut:
<a id='tag'></a> [something pointer](#tag)

# Table of Contents (ongoing)
1. [data wrangling](#dw)
      - [read file](#rf)
3. [model](#model)
      - [model training](#mt)
      - [multi GPUs](#mgpu)
      - [distributed data parallel](#ddp)
      - [small tricks](#st)
5. [utils](#utils)
6. [advanced technique in ML](#atim)
7. [great notebook](#gn)

# Data Wrangling
<a id='dw'></a>
transform raw data into useable format
### Read File
<a id='rf'></a>
- my script - [read csv one by one and clean at same time](https://github.com/tinghe14/Reusable-Code/blob/ab1f36b3db68cadbfe04f88b88bda2471168c743/Data%20Wrangling/Read%20File/0.py)


# Model
<a id='model'></a>
build, train, evaluate model
### Model Training
<a id='mt'></a>
- my script - [local script example](https://github.com/tinghe14/Reusable-Code/tree/92b083ab226d24dfc3dc75fabee77a3623789180/Model/Model%20Training/%20Local%20Script)
  - contains scripts from pre-cleaning, training and validation, testing, config files and apply early stopping and saving and loading checkpoints, training on gpu and cpu locally
- my script - use argparse package instead of config: easy to run in the terminal
  - my script - [add small trick to be able to run at Colab](): (1)add -f argument (2)copy file from gdrive to current colab temporary directory (3)copy the task.py to main.ipynb and run on the main.ipynb
- [GCP version]()
### Multi GPUs
<a id='mgpu'></a>
- [multi GPUs]
### Distributed Data Parallel
<a id='ddp'></a>
- [Other blog in Chinese - 分布式训练 - 多机多卡](https://blog.csdn.net/love1005lin/article/details/116456422)
### Small Tricks
<a id='st'></a>
- my script - [breif summary of training parametes in NN](https://github.com/tinghe14/Reusable-Code/blob/db1f7eb6aa37d3bc138d617e4fdbe6e76f1a53cd/Model/Small%20Trick/Summary%20Param/0_summary_parameter.py)
- my note - [error: expect to be on the same device](https://github.com/tinghe14/Reusable-Code/blob/6e5c5588a6dbb5f26fcc3a6518d255bdea6d1df8/Model/Small%20Trick/Expect%20Same%20Device/0_expect_same_device.md)

# Utils
<a id='utils'></a>
general functions which not involve any bussiness logic
- other package: [generate tree of files under directory](https://github.com/michalbe/md-file-tree)
- my script: [check folder if not create one](https://github.com/tinghe14/Reusable-Code/blob/c7bec6ed44d27782f882e4a9b33c80f8cd0dcc2d/Utils/Make%20Dir/0_make_direct.py)
- my script - [generate requirements txt file](https://github.com/tinghe14/Reusable-Code/blob/61768a95e5cd9a65bbd757addd676725803607ed/Data%20Wrangling/Make%20Req%20File/0.py)

# Advanced Technique in ML
### Ensemble 
- [Huge Ensemble](https://www.kaggle.com/code/thedevastator/huge-ensemble)

# Great Notebook
- kaggle notebook grandmaster: https://www.kaggle.com/thedevastator

<!---
https://www.1point3acres.com/bbs/thread-997815-1-1.html
- 现在市场上有好多找做LLM背景人的坑。
我好奇这样背景的人和普通做NLP的人有什么主要的差异吗？
例如我这样的水货背景
- 3年前搞过一点NLP，会做常见的一些task（分类、问答、翻译什么的）。最近几年的进展都没怎么跟了。
- 明白古早版本的bert，transformer，gpt都是怎么工作的。
- 知道language model是怎么弄出来的（large的没碰过）
- 知道多机多卡的训练怎么写
- 会用一些已有推理框架onnx，tensorrt什么的捣鼓捣鼓模型上线
我可以大言不惭的说自己也是LLM背景的人吗？还是会被打回原型？
可能lz的能力能应付大多数工作了，但不足以在众多简历中被选出来，因为这些东西很多人都会。属实，感觉自己只能算个民科。研究方面完全没碰过。
很好的讨论，现在的公司精得很，感觉有没有百亿到千亿param 模型的实战的经验很容易就能在面试中看出来，在lz的基础上分享一些最近半年和相关资方打交道感受到的他们的期望和standard：
- 3年前搞过一点NLP，会做常见的一些task（分类、问答、翻译什么的）。最近几年的进展都没怎么跟了。
  --是否知道用10B以上LLM怎样便宜又有效的实现这些应用，LLM+RLHF/prompt engineering相比传统bert做基础任务有怎样的pros cons，怎样增强robustness/fairness
- 明白古早版本的bert，transformer，gpt都是怎么工作的
   --是否能在面试时不查api的情况下半小时pytorch/tf手撸朴素的bert/gpt实现 从 tokenizaiton/embedding/self attention and ffn 到beam search?
- 知道language model是怎么弄出来的（large的没碰过）
  --千亿规模模型训练都有哪些坑，数据清洗去重有哪些坑和调优技巧？怎么通过各种training dynamics的参数寻找适合的训练参数和训练早期发现不适合的模型参数？
- 知道多机多卡的训练怎么写.
   --megatron实现代码是否熟悉，知道如何修改？pipeline/tensor/data parallelism各项参数应该如何配置
- 会用一些已有推理框架onnx，tensorrt什么的捣鼓捣鼓模型上线
  --onnx/tensorrt/triton/pytorch2.0/deepspeed/fastertransformer用来部署百亿以上模型各有什么坑，如果需要4bit、8bit部署怎样为这些还不支持int8/int4实现相应的cuda kernel并调优超过cublas的水平？
可能他们进的早，我最近面openai和anthropic一类的公司 被问的比刚才列的还深
哎 确实有些面试造火箭的感觉 谁让现在这领域卷呢 不过倒也不用都精通，在一个方面比较专，其他方面能说出一些思考就行
我觉得偏工程的关心也没那么多
除了那几个Transformer的model外 (可以去Huggingface看) 也就是deepspeed zero了 ..... 我只会用data parallel 最多搞30-40B model 需要model/pipeline parallel 我也不知道哪个好
偏研究的东西就比较多了 最好还是经常看论文
比如比较新的positional encoding -> alibi / rotary 这种 会被考到
- 怎么说呢，比 LZ 水的搞 LLM 的人也有，比 LZ 强的面试进不去的也有。
- LZ 是想去搞 LLM，或者说是想去 OpenAI/Google Bard 这种吗？如果不是下面的建议不用看。
- 建议 LZ 跳出学生思维：不是我会这个技术，我就能去搞。
- 想明白这一点：你能为别人贡献什么，别人为什么需要你？
--->
