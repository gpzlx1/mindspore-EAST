yes | sudo cp /usr/lib64/libpython3.7m.so.1.0.os-origin /usr/lib64/libpython3.7m.so.1.0

# 安装三方python包需要的编译依赖
#sudo yum install geos-deve
export LD_LIBRARY_PATH=/home/work/user-job-dir/EAST/lib:$LD_LIBRARY_PATH

yes | sudo cp /usr/lib64/libpython3.7m.so.1.0.compiled-3.7.5 /usr/lib64/libpython3.7m.so.1.0

# 安装三方Python库
pip install shapely