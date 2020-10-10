Matlab implementation of the MBGD-RDA algorithm and its variants in the following paper of TSK fuzzy systems for regression problems:

D. Wu*, Y. Yuan, J. Huang and Y. Tan*, "Optimize TSK Fuzzy Systems for Regression Problems: Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. on Fuzzy Systems, 28(5), pp. 1003-1015, 2020.

By Dongrui Wu, drwu@hust.edu.cn

main.m: Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.

MBGD_RDA: Implementation of the MBGD-RDA algorithm in the above paper, using Gaussian MFs, and specifies the number of Gaussian MFs in each input domain.

MBGD_RDA_T: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using trapezoidal MFs, and specifies the number of trapezoidal MFs in each input domain. The derivations can be found in derivation.pdf.

MBGD_RDA2: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using Gaussian MFs, and specifies the total number of rules. It's more flexible than MBGD_RDA, and usually performs better. The derivations can be found in derivation.pdf.

MBGD_RDA2_T: Implementation of a variant of the MBGD-RDA algorithm in the above paper, using trapezoidal MFs, and specifies the total number of rules. It's more flexible than MBGD_RDA_T, and usually performs better. The derivations can be found in derivation.pdf.
