# gpu_miner
CUDA GPU miner for BTCW

You can use the same wallet.dat for all GPUs. You must tell the GPU to use a different stage2 selection number to ensure unique stage2 hashing when using the same stage1 keys.

GPU miner only currently works for LINUX and uses shared memory.

DO NOT start the BTCW GPU miner until the BTCW node has started its mining process.

start the gpu miner for gpu number 1
```
./gpu_miner 1
```


start the gpu miner for gpu number 2
```
./gpu_miner 2
```

start the gpu miner for gpu number 3
```
./gpu_miner 3
```

start the gpu miner for gpu number 1000
```
./gpu_miner 1000
```