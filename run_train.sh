mkdir -p log

max=6
for i in `seq 1 $max`
do
    python dialog_pytorch.py -task_id "$i" > train_log"$i"
done