for i in {1..15};
do
	echo $i
	CUDA_VISIBLE_DEVICES=$1 python3 main_ddis.py --class_index $i --train True --evaluate False
done
