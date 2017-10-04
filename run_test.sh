# python s
# python dialog_pytorch.py -task_id 1
mkdir -p log

max=6
for i in `seq 1 $max`
do
    python dialog_pytorch.py -task_id "$i" -train False > log/test_log"$i"
done