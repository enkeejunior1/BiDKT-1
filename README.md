# KT-BERT

The this repo is about developing KT-BERT which is used BERT to knowledge tracing tasks.

# How to start?

1. You need a datasets folder; the path is ./datasets
2. You can use the preprocessed datasets. See the Dataset Section
3. If you want to train the model, you can use scrip like this.
   """
   python train.py --model_fn model.pth --model_name bert4kt_plus --dataset_name assist2009_pid

   """
4. If you need to other options, please see the define_argparser.py.
5. If model_name contain the "_plus", you must train that model with pid_loader; assist2009_pid, assist2017_pid...

# Requirements

We used docker image, 'ufoym/deepo' and used some other packages.

If you don't use docker, you can use requirements.txt.

# Dataset

We ignored students datasets under 5 interaction.

You can download the datasets by using link below.

https://drive.google.com/drive/folders/1B5vHabKUwzGEQAyDaj_0cBT6UqTMA13L?usp=sharing

If you have any question, please send me a email.

codingchild@korea.ac.kr
