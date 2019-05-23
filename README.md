# pytorch-stacked-hourglass

This is a fork of [bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose) which is modified
for use as a Python package.

## Evaluation scripts

Generate predictions file (PCKh score printed is not the proper reported value):

```
$ python example/main.py --dataset mpii \
    --anno-path data/mpii/mpii_annotations.json \
    --image-path /data/datasets/MPII_Human_Pose/images \
    -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/mpii/hg_s2_b1 \
    --resume checkpoint/mpii/hg_s2_b1/model_best.pth.tar -e
```

Calculate proper PCKh:

```
$ python evaluation/eval_PCKh.py -r checkpoint/mpii/hg_s2_b1/preds_valid.mat
```
