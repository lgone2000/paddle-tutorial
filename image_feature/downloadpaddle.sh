wget http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl
alias pip="~/easy-paddle-run/paddle-release/python-gcc482-paddle/bin/python /home/users/liuguoyi01/easy-paddle-run/paddle-release/python-gcc482-paddle/bin/pip"
pip uninstall -y paddlepaddle-gpu
pip install paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl

