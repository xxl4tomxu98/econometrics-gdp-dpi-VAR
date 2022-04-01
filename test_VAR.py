import os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats

#Summary Statistics
def check_summary(series):
    """
    Split the series into two and check summary statistics for both the series
    """
    
    X = series.values
    split =int( len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    meanoverall, mean1, mean2 = X.mean(), X1.mean(), X2.mean()
    varoverall, var1, var2 = X.var(), X1.var(), X2.var()
    print('mean overall = %f, \t mean1=%f, \t mean2=%f' % (meanoverall, mean1, mean2))
    print('variance overall= %f, \t variance1=%f, \t variance2=%f' % (varoverall, var1, var2))


#Distribution by plotting Histogram
def check_distribution(series):
    """
    Distribution by plotting Histogram
    """
    series.hist()
    plt.show()        


#Stationarity Check:(ADF)
def check_stationarity(series):   
    """
    Check stationarity using ADF test
    Ho : The series is non-stationary and has a unit root --> p-value > 0.05
    H1 : The series is stationary and does not have a unit root --> p-value < 0.05
    """
   
    X = series.values
    result = adfuller(X)
    print('Test: Running ADF Stationarity test for {} \n\t ADF test statistic: {} \n\t p-value: {} '.format(series.name, result[0], result[1]))

    print('\n\tCritical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
    
    if result[4]['5%'] > result[0]:
        conclusion = 'Series is Stationary @95% CI'
        print ('\t Conclusion: Series is stationary at 95% CI\n')
    elif result[4]['10%'] > result[0]:
        conclusion = 'Series is Stationary @90% CI'
        print ('\t Conclusion: Series is stationary at 90% CI but non-stationary at 95% CI\n')
    else:
        conclusion = 'Series is non-Stationary'
        print ('\t Conclusion: Series is non-stationary\n')
    return conclusion
        
#Series Plots:
def check_plots(series):
    """
    Plots the series as a line chart
    """
    plt.plot(series)
    plt.show()        
    
def check_normality(series):
    result = stats.normaltest(series)
    
    print('Test: Running Agostino and Pearson Normality test for {} \n\t Normality test statistic: {} \n\t p-value: {} '.format(series.name, result[0], result[1]))
    
    if(result[1]< 0.05):
        conclusion = 'Not Normally Distributed'
        print('\t Conclusion: {} is not normally distributed at 95% CI \n'.format(series.name))
    else:
        conclusion = 'Normally Distributed'
        print('\t Conclusion: {} is normally distributed at 95% CI \n'.format(series.name))
        
    return conclusion