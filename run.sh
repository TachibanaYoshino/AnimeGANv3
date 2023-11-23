nohup  python train.py  > output.log 2>&1 &

sleep 3

tail -f output.log

# CUDA_VISIBLE_DEVICES=0,1,2,3
# ps -ef |grep joblib |grep -v grep |awk '{print "kill -9 "$2}' | sh