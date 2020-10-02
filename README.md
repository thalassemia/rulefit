# rulefit

**Full Disclaimer**: This is based heavily on the original Python RuleFit implementation by Christopher Molnar (check out [his Github repo](https://github.com/christophM/rulefit)).

The main tweaks that I've made are listed below:
1. Uses LightGBM instead of scikit-learn's built-in tree ensemble functions
2. Uses scikit-learn's stochastic gradient descent functions (SGDRegressor/SGDClassifier) instead of LassoCV/LogisticRegressionCV

In my limited testing, the first tweak significantly improves the achievable accuracy while the second drastically reduces training time.

Note that the default parameters are tuned for my specific use case and will likely need to be modified for other data.
