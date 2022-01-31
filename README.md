# XGBoost C++ Playground

This repo shows how to use XGBoost in C++. While there is no officially supported [C++ API](https://xgboost.readthedocs.io/en/stable/c%2B%2B.html), it is possible to directly call the internal C++ functions without using the [C API](https://xgboost.readthedocs.io/en/stable/c.html).

To get started, download the url dataset [from kaggle](https://www.kaggle.com/br0kej/feature-engineering-and-analysis/data?select=malicious_phish.csv) and save it to the `data` directory. Then, execute the `urls.ipynb` once. It will save a split and feature engineered version of the dataset to the `data` directory. While running it, pay attention to the confusion matrix the python code has generated. Then, run `urls.cc`, it will produce the same result as the python code.

If you want to extend the C++ code, I suggest you take a look at the implementation of the C functions in [`include/xgboost/c_api.h`](https://github.com/dmlc/xgboost/blob/master/include/xgboost/c_api.h).