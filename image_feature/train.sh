export GLOG_logtostderr=1
export GLOG_minloglevel=0
export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64/:/home/work/cudnn/cudnn_v7/cuda/lib64/:/home/liuguoyi01/work/easy-paddle-run/paddle-release/lib
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1

#env CUDA_VISIBLE_DEVICES=0,1,2,3 python traincifar.py
env CUDA_VISIBLE_DEVICES=0,1,2,3 python trainface.py