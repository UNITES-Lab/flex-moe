device=0
modality=IGCB
lr=1e-4
num_experts=16
num_layers_fus=1
top_k=4
train_epochs=50
warm_up_epochs=5
hidden_dim=128
batch_size=8
num_heads=4

CUDA_VISIBLE_DEVICES=$device python main.py \
    --n_runs 3 \
    --modality $modality \
    --lr $lr \
    --num_experts $num_experts \
    --num_layers_fus $num_layers_fus \
    --top_k $top_k \
    --train_epochs $train_epochs \
    --warm_up_epochs $warm_up_epochs \
    --hidden_dim $hidden_dim \
    --batch_size $batch_size \
    --num_heads $num_heads