## Our results (test split)

| model                          | family    |   accuracy |   off_by_one |   rmse |   mae |   macro_f1 |   recon_auc |
|:-------------------------------|:----------|-----------:|-------------:|-------:|------:|-----------:|------------:|
| resnet34_adamw_onecycle_cw_tta | cnn       |      0.445 |        0.838 |  1.109 | 0.754 |      0.534 |       0.727 |
| resnet34_sgd_sgdr_tta          | cnn       |      0.424 |        0.839 |  1.092 | 0.763 |      0.505 |       0.705 |
| resnet18_sgd_step_tta          | cnn       |      0.406 |        0.840 |  1.099 | 0.780 |      0.497 |       0.699 |
| svm_rbf [hog+color+blob]       | classical |      0.306 |        0.736 |  1.353 | 1.024 |      0.390 |     nan     |
| svm_rbf [hog_edge]             | classical |      0.306 |        0.673 |  1.549 | 1.158 |      0.358 |     nan     |
| svm_rbf [hog]                  | classical |      0.294 |        0.719 |  1.408 | 1.070 |      0.354 |     nan     |
| svm_rbf [raw]                  | classical |      0.270 |        0.724 |  1.330 | 1.046 |      0.265 |     nan     |
| svm_linear [raw]               | classical |      0.253 |        0.628 |  1.714 | 1.313 |      0.277 |     nan     |
| logreg [raw]                   | classical |      0.252 |        0.629 |  1.691 | 1.303 |      0.286 |     nan     |
| svm_rbf [blob]                 | classical |      0.237 |        0.612 |  1.758 | 1.364 |      0.246 |     nan     |
| tree [raw]                     | classical |      0.234 |        0.574 |  1.821 | 1.425 |      0.264 |     nan     |

## Reference paper (reported, for context only)

| model                             |   accuracy |    rmse |
|:----------------------------------|-----------:|--------:|
| [paper] SVM RBF + PCA (150K imgs) |      0.320 | nan     |
| [paper] ResNet34 SGDR (150K imgs) |      0.538 |   0.940 |
| [paper] ResNet34 Adam (324K imgs) |      0.562 |   0.900 |

Random baseline for 6 classes: 0.167
