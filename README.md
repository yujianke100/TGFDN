# TGFDN

## QueryOPT
The project that helps us implement (alpha, beta)-core is [QueryOPT](https://github.com/boge-liu/alpha-beta-core).

## Quick Start

1. download the dataset from the [competition](https://tianchi.aliyun.com/dataset/dataDetail?dataId=123862). Please download the final round dataset.
2. Pre-process: ``python dataset/get_data.py``
3. Install [swig](https://github.com/swig/)
4. Build pyabcore: ``sudo apt-get -y install libboost-all-dev && cd ./queryopt && ./build.sh && cd ..``
5. Start:``python main.py``

## Note
The last four entries of customer vertex in the dataset are noise and have been ignored. Experimental results may vary slightly due to different hardware configurations.
