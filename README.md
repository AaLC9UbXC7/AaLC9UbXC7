Codes and Dataset for "Accessing Student Performance with Multi-granularity Attention from Online Classroom Dialogue"

## DataSet

We publish the datasets of our experiments. The dataset contains the raw teacher's and student's conversations in each example question. And also contain the data after preprocessing, which can be straight used to train models.

The datasets can download from [here](https://drive.google.com/drive/folders/1o4FqtRsmWMq80adqC9eFu6A0cfpXTIac?usp=sharing) after downloading the data put the data in `data` folder. The folder structure is following:

`data`

- `dataset`
- `features`
- `word2vec`
- `README.md`

The dataset description can see in [there](data/README.md)

## Create conda env

Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`

## Codes

You can get the results in our paper by running the following command. All codes are in `codes` folder.

### Requirements

Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`

### Training

**GBDT**

See `GBDT.ipynb`

**HAN**

```shell
python han.py --emb_name=word2vec --lr=0.001 --mode=static --seed=2022 --word_num_hidden=32
```

**BERT**

```shell
python bert.py --area_attention=0 --emb_name=edu_roberta_cls --lr=0.0001 --num_attention_heads=8 --num_hidden_layers=1 --output_dense_num=2 --seed=421
```

**LLaMA-7B**

```shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=21009 \
    train_llama.py \
    --model_name_or_path {model_name_or_path} \
    --data_path {data_path} \
    --bf16 True \
    --output_dir {output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 38 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

**Chinese-LLaMA-7B**

```shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=21009 \
    train_llama.py \
    --model_name_or_path {cLLaMA_model_name_or_path} \
    --data_path {data_path} \
    --bf16 True \
    --output_dir {output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 38 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

**Ours**

```shell
python lstm_gate.py --area_attention=1 --deep_dnn_dropput=0.1 --deep_dnn_num=1 --deep_output_pooling_mode=se_att --lr=0.0001 --lstm_layer_num=2 --lstm_output_dim=256 --max_area_width=3 --seed=3407 --use_attention=1
```
