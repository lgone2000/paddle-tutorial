export GLOG_logtostderr=1
export GLOG_minloglevel=0
export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64/:/home/work/cudnn/cudnn_v7/cuda/lib64/:/home/liuguoyi01/work/easy-paddle-run/paddle-release/lib:./lib
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
export CPU_NUM=4
export FLAGS_fraction_of_gpu_memory_to_use=0.2
# sudo fuser -v /dev/nvidia*
#alloc 50% gpu memory for train. usefull when other programs running on gpu
export  CUDA_VISIBLE_DEVICES=0,1,2,3 

rm -fr tempfacemodel
python inferface.py test_convertsnap2inference $1 tempfacemodel 3,112,112 512
python inferface.py eval_face tempfacemodel
