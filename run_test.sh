# python s
# python dialog_pytorch.py -task_id 1
max=6
for i in `seq 1 $max`
do
    python dialog_pytorch.py -task_id "$i" -train False > test_log"$i"
done