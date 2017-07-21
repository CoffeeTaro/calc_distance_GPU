#!/bin/bash

nvcc -c -O3 calculateDistance.cu -o calc_distance.o -Xcompiler -fPIC -arch=sm_30
gcc -pthread -shared -L/home/kotaro/.pyenv/versions/anaconda3-4.4.0/lib -Wl,-rpath=/home/kotaro/.pyenv/versions/anaconda3-4.4.0/lib,--no-as-needed calc_distance.o -L/home/kotaro/.pyenv/versions/anaconda3-4.4.0/lib -lpython3.6m -o calc_distance_gpu.so
