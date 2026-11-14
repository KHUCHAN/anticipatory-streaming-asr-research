# Epoch-wise double-descent investigation

Status: in progress. Epochs 0–27 are inherited; only epochs 28–96 are newly trained here. No overfit-based stopping. Three warm-started depths and one seed do not establish a model-wise double-descent curve.

# FGN capacity study — current status

Fixed-case fitting diagnostic; no test use. Residual correction/encoder skipping is not implemented.

| FGN blocks | Generator parameters | Epochs | Best epoch | Best dev NLL | Last train-probe NLL | Last dev NLL | Overfit confirmed |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 6,323,968 | 70 | 5 | 0.9486 | 0.1134 | 1.2910 | Yes |
| 8 | 10,535,680 | 27 | 5 | 0.9509 | 0.1322 | 1.1269 | Yes |
| 16 | 18,959,104 | 96 | 5 | 0.9504 | 0.1272 | 1.4575 | Yes |

Matched-epoch comparison at epoch 27:

| FGN blocks | Train-probe NLL | Dev NLL | Train-probe objective | Dev objective |
|---|---:|---:|---:|---:|
| 4 | 0.1313 | 1.1147 | 1.1448 | 1.9399 |
| 8 | 0.1322 | 1.1269 | 1.1449 | 1.9491 |
| 16 | 0.1319 | 1.1306 | 1.1447 | 1.9516 |

A completed stage without the predeclared pattern is not evidence of overfitting. Online loss is the unweighted mean of training-batch objectives during each epoch; fixed-probe objectives use evaluation mode after that epoch. They differ in both examples and model state. Neither is unsmoothed token NLL.

![Current learning curves](learning_curves.png)
