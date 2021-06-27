# Multivariate Time Series Analysis

A univariate time series data contains only one single time-dependent variable while a multivariate time series data consists of multiple time-dependent variables. We generally use multivariate time series analysis to model and explain the interesting interdependencies and co-movements among the variables. In the multivariate analysis — the assumption is that the time-dependent variables not only depend on their past values but also show dependency between them. Multivariate time series models leverage the dependencies to provide more reliable and accurate forecasts for a specific given data, though the univariate analysis outperforms multivariate in general[1]. In this article, we apply a multivariate time series method, called Vector Auto Regression (VAR) on a real-world dataset.

# Vector Auto Regression (VAR)

VAR model is a stochastic process that represents a group of time-dependent variables as a linear function of their own past values and the past values of all the other variables in the group.
For instance, we can consider a bivariate time series analysis that describes a relationship between hourly temperature and wind speed as a function of past values [2]:
temp(t) = a1 + w11* temp(t-1) + w12* wind(t-1) + e1(t-1)
wind(t) = a2 + w21* temp(t-1) + w22*wind(t-1) +e2(t-1)
where a1 and a2 are constants; w11, w12, w21, and w22 are the coefficients; e1 and e2 are the error terms.

# Dataset

[Statmodels](https://www.statsmodels.org/stable/index.html) is a python API that allows users to explore data, estimate statistical models, and perform statistical tests. It contains time series data as well. We download a dataset from the API.
