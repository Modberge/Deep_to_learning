
This is the repository for Zhejiang university "AI safe"(2021 autumn) homewor#1.

At the same time, it is also the repository for self-study the book "DIVE INTO DEEP LEARNING" 

## Deep Neural Network
 Pytorch

## Version
 1.9+cu111

## Dataset
 CIFAR10

## To run this homework and get result:

```
cd ch_5
python ResNet.py
```

## Result

### Setting batch_size

| batch_size |  32  |  64  |  128 |  256 | 1024 |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Accuracy   |0.7879|0.7672|0.7918|0.7685|0.7839|

| learn_rate |  1e0 | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Accuracy   |0.1000|0.3323|0.4822|0.7099|0.6782|

| num_epoch  |  1   |   5  |  10  |  15  |  20  |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Accuracy   |0.5992|0.6764|0.6281|0.6816|0.7179|

|    decay    | 1e0 | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Accuracy   |0.6222|0.7276|0.6787|0.6698|0.7132|
