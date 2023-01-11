# Codes for acl2023

## Create conda env  
Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`

## DataSet 

We publish the datasets of our experiments. The dataset contains the raw teacher's and student's conversations in each example question. And also contain the data after preprocessing, which can be straight used to train models.

The datasets can download from [this url](https://drive.google.com/file/d/1dUHYLKoE09Y8D5I0v4x77m9zIJECaRhI/view?usp=share_link), after downloading the data put the data in `data` folder. The folder structure is following:

`data`
- `dataset`
- `features`
- `word2vec`
- `README.md`

The dataset description can see in [there](data/README.md)



## Training

You can get the results in our paper by running the following command.

**HAN**

```python
python han.py --emb_name=word2vec --lr=0.001 --mode=static --seed=2022 --word_num_hidden=32
```

**BERT**

```python
python bert.py --area_attention=0 --emb_name=edu_roberta_cls --lr=0.0001 --num_attention_heads=8 --num_hidden_layers=1 --output_dense_num=2 --seed=421
```

**Ours**
```python
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=se_att --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=1
```


