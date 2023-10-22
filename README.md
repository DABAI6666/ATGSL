# ATGSL
Adversarial Text Generation by Search and Learning




## Data
Our 7 datasets are:

## Prerequisites:
Required packages are listed in the requirements.txt file:
```
pip install -r requirements.txt
```

## How to use


* Run the following code to generate the adversaries:

```
python command_dataset_model.py
```




Here we explain each required argument in details:

  * --data_set: The path to the dataset. We used the public datasets [datasets](https://drive.google.com/drive/folders/1kYuCDtt2Tvtw5OR1ccdUlw_IxFCQSvtD?usp=sharing).
  * --victim: Name of the target model such as ''bert''. [trained model parameters](https://drive.google.com/drive/folders/1dhGC4wrqPBCQsBrTRTkJmF7rNBPPE9S9?usp=sharing).
  * --bm_model: Our generated finely tuned BM_Model,we shared the [trained model parameters](https://drive.google.com/drive/folders/1Sy6G3EWSeK-phryB82pMQDBMbwDBdR3p?usp=sharing).
  * --train_classification: Used to generate victim models.
  * --model: All core model codes of ATGSL.




In case someone may want to use our generated adversary results towards the benchmark data directly, [here it is](https://drive.google.com/drive/folders/1yZ_wmI6YHEhRgTJw83KL1dgN-fEJ7bwK?usp=sharing).
# Hadoop
