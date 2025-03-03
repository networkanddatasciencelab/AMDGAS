#!/bin/bash
cd /home/mx/Desktop/nmgas/Comm
source /home/mx/miniconda3/etc/profile.d/conda.sh
conda activate d2l

# 定义变量
lr="5e-3"
weight_decay="5e-4"
epochs="1001"
nhid="16"
nout="10"
NMF_feature_num="20"
is_draw="False"


train_name="Cora"
# 更改模型名称为 GSACN 并重复实验
model_name="GSACN"
# 使用循环和变量来运行和记录实验
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GCN 并重复实验
model_name="GCN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GAT 并重复实验
model_name="GAT"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GIN 并重复实验
model_name="GIN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 SAGE 并重复实验
model_name="SAGE"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GraphConv 并重复实验
model_name="GraphConv"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 MLP 并重复实验
model_name="MLP"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

train_name="Citeseer"
# 更改模型名称为 GSACN 并重复实验
model_name="GSACN"
# 使用循环和变量来运行和记录实验
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GCN 并重复实验
model_name="GCN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GAT 并重复实验
model_name="GAT"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GIN 并重复实验
model_name="GIN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 SAGE 并重复实验
model_name="SAGE"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GraphConv 并重复实验
model_name="GraphConv"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 MLP 并重复实验
model_name="MLP"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done


train_name="Pubmed"
# 更改模型名称为 GSACN 并重复实验
model_name="GSACN"
# 使用循环和变量来运行和记录实验
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GCN 并重复实验
model_name="GCN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GAT 并重复实验
model_name="GAT"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GIN 并重复实验
model_name="GIN"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 SAGE 并重复实验
model_name="SAGE"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 GraphConv 并重复实验
model_name="GraphConv"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done

# 更改模型名称为 MLP 并重复实验
model_name="MLP"
for seed in {4215..4219}; do
    log_file="./log/${train_name}_${model_name}_seed=${seed}.log"
    nohup stdbuf -oL python gs_comm.py --seed ${seed} --train_name "${train_name}" --model_name "${model_name}" --lr ${lr} --weight_decay ${weight_decay} --epochs ${epochs} --nhid ${nhid} --nout ${nout} --NMF_feature_num ${NMF_feature_num} --is_draw ${is_draw} > ${log_file} &
    wait
done




