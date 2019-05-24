# pytorch-stacked-hourglass

This is a fork of [bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose) which is modified
for use as a Python package.

## Evaluation scripts

Here's a quick example of evaluating the pretrained 2-stack hourglass model
[from Google Drive](https://drive.google.com/drive/folders/0B63t5HSgY4SQTzNQWWplelF3eEk).

```bash
$ python example/main.py --dataset mpii \
    --image-path /data/datasets/MPII_Human_Pose/images \
    -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/mpii/hg_s2_b1 \
    --resume checkpoint/mpii/hg_s2_b1/model_best.pth.tar -e
```

Output:

```
==> creating model 'hg', stacks=2, blocks=1
=> loading checkpoint 'checkpoint/mpii/hg_s2_b1/model_best.pth.tar'
=> loaded checkpoint 'checkpoint/mpii/hg_s2_b1/model_best.pth.tar' (epoch 185)
    Total params: 6.73M

Evaluation only
Eval  |################################| (493/493) Data: 0.145060s | Batch: 0.399s | Total: 0:03:16 | ETA: 0:00:01 | Loss: 0.0002 | Acc:  0.8458

PCKh scores following proper evaluation protocol:
Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean
96.15  94.89     88.14  83.78   87.43   82.19   77.87   87.33
```
