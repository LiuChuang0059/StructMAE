dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="MUTAG"
[ -z "${device}" ] && device=-1

python main_graph.py \
	--device $device \
	--dataset $dataset \
	--encoder "gin" \
	--decoder "gin" \
	--weight_decay 0.0 \
	--optimizer adam \
	--drop_edge_rate 0.0 \
	--loss_fn "sce" \
	--activation "prelu" \
	--norm "batchnorm" \
	--seeds 0 1 2 3 4 \
	--use_cfg \
