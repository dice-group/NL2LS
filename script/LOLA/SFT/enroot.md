To start interactive GPU session:
```bash
NGPU=1 # Number of GPUs
TIME=01:00:00 # 1 hour
CORES_PER_GPU=16
CORES=$(($CORES_PER_GPU * $NGPU))
srun -N 1 -n 1 -c $CORES --gres=gpu:$NGPU -t $TIME --pty bash
```

To create container:
```bash
enroot create --name nl2ls nl2ls.sqsh
```

To start container:
```bash
enroot start -r -w nl2ls
```


To run a script in SPTQA inside container with already mounted directory:
```bash
srun -N 1 -n 1 -c 1 --gres=gpu:1 -t 00:01:00 --container-image ./nl2ls.sqsh touch /NL2LS/script/LOLA/SFT/testfile
```
