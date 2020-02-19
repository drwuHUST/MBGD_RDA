Matlab implementation of the MBGD-RDA algorithm and its variants in the following paper of TSK fuzzy systems for regression problems:

Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. on Fuzzy Systems, 2020, accepted.

By Dongrui Wu, drwu@hust.edu.cn

main.m: Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.

MBGD_RDA: Implementation of the MBGD-RDA algorithm in the above paper, using Gaussian MFs, and specifies the number of Gaussian MFs in each input domain.

MBGD_RDA_T: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using trapezoidal MFs, and specifies the number of trapezoidal MFs in each input domain. The derivations can be found in derivation.pdf.

MBGD_RDA2: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using Gaussian MFs, and specifies the total number of rules. It's more flexible than MBGD_RDA, and usually performs better. The derivations can be found in derivation.pdf.

MBGD_RDA2_T: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using trapezoidal MFs, and specifies the total number of rules. It's more flexible than MBGD_RDA_T, and usually performs better. The derivations can be found in derivation.pdf.
