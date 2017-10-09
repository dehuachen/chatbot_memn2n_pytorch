mkdir -p log

max=6
for i in `seq 1 $max`
do
    echo "running taks $i"
    python dialog_pytorch.py -task_id "$i" > log/train_log"$i"
    echo "finished taks $i"
done
