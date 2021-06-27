from test-VAR import *
from statsmodels.tsa.api import VAR, DynamicVAR
import statsmodels.api as sm
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.vector_ar import plotting
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot

#########
# Load pre-loaded macroeconomic data from PANDAS
#########

mdata = sm.datasets.macrodata.load_pandas().data

 # prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)

mdata1 = mdata[['realgdp', 'cpi', 'unemp', 'infl']]
mdata1.index = pd.DatetimeIndex(quarterly)

"""
a. Take Log difference of the level variables
b. Take differences of the rate variables
"""
mdata1['realgdp_logdiff'] = pd.Series(np.log(mdata1['realgdp']).diff().dropna())
mdata1['cpi_logdiff'] = pd.Series(np.log(mdata1['cpi']).diff().dropna())

mdata1['unemp_diff'] = pd.Series(mdata1['unemp'].diff().dropna())
mdata1['infl_diff'] = pd.Series(mdata1['infl'].diff().dropna())

"""
a. Drop NA
"""
mdata1.dropna(inplace = True)


"""
a. Basic checks
"""
#Normal line plots
check_plots(mdata1['realgdp'])    
check_plots(mdata1['unemp'])    
check_plots(mdata1['unemp'].diff().dropna())
    
#Summary Statistics
check_summary(mdata1['cpi_logdiff'])    
check_summary(mdata1['unemp'])

#Distribution by plotting Histogram
check_distribution(mdata1['cpi_logdiff'])
check_distribution(mdata1['unemp'])
check_distribution(mdata1['unemp'].diff().dropna())

#Stationarity Check:  
check_stationarity(mdata1['realgdp'])
check_stationarity(mdata1['realgdp_logdiff'])
check_stationarity(mdata1['infl'])
check_stationarity(mdata1['unemp'].diff().dropna())


"""
a. Prepare modeling dataset
"""
temp_data = pd.DataFrame(mdata1).dropna()
model_data = temp_data[['unemp_diff', 'realgdp_logdiff', 'cpi_logdiff']]


"""
a. Create VAR models -        
"""

def make_var_model(data, lags = 1, actual_plot = False ):
 
    # make a VAR model
    model = VAR(data)
    
    result_dict = {}
    
    writer = pd.ExcelWriter('Models.xlsx', engine='xlsxwriter')
        
    for lag in range(1, lags+1):
        #Fitting Model
        results = model.fit(maxlags = lag)
        
        lag_order = results.k_ar
        
        print ('Exogenous Variables for the model with Lag: %d \n '%lag+ str(results.exog_names))
        print (results.summary())
        
        #Generating Model output
        fitted_values = results.fittedvalues
        forecast_values = pd.DataFrame(data = results.forecast(y= data.values[-lag_order:], steps=  5), columns = results.names)
        results.forecast_interval(y= data.values[-lag_order:], steps = 10)
        
        results.plot_forecast(steps = 10, plot_stderr = True)
        
        if actual_plot ==True:
            results.plot()

        #Storing Residuals for testing purpose
        residuals = results.resid.add_prefix('Residuals_')
        normality_var = {}
        stationarity_var = {}
        acf_data = {}
        
        print("************* Running Stationarity and Normality for the Residuals *************\n")  
        for column in residuals.columns:
            #test for normality of residuals
            normality_var[column] = check_normality(residuals[column])
            #test for stationarity of residuals
            stationarity_var[column] = check_stationarity(residuals[column])
            #acf plots of residuals of each variable for 10 lags
            acf_data[column] = acf(residuals[column])[0:10]       
        
        print("************* Plotting ACF Plots for the Residuals *************\n")   
        pd.DataFrame(data = acf_data).plot(kind = "bar", title = 'ACF plots for the residuals Lag_Order_{}'.format(lag_order))
        plt.savefig('ACF_Plot_Lag_Order_{}.png'.format(lag_order))
            
        equation_model_data = pd.DataFrame(index = results.tvalues.index)

        #writing results of equations to different excel sheets
        for var, column in enumerate(data.columns):
            print('Writing in excel for variable {}'.format(column))
            sheet =  "lag" + str(lag_order)
            coefs = []
            for lag in range(lag_order):
                coefs.append(results.coefs[lag][var])
            intercept = np.asarray(results.intercept[var])
            equation_model_data["coefs"] = np.append(intercept, coefs)
            equation_model_data["std err"] = results.stderr[column]
            equation_model_data["t-values"] = results.tvalues[column]
            equation_model_data["p-values"] = results.pvalues[column]
            equation_model_data["stationarity check on Residuals"] = stationarity_var['Residuals_'+column]
            equation_model_data["normality check on Residuals"] = normality_var['Residuals_'+column]
            equation_model_data.to_excel(writer, sheet_name = sheet, startrow = var*10, startcol=0)
            
            print('Results written to excel file')        
        
        result_dict['Lag_Order_{}'.format(lag_order)] = equation_model_data
                   
    return result_dict

#make_var_model(data = mdata1, lags = 1, actual_plot = False )
    
Results = make_var_model(data = model_data, lags = 2, actual_plot = False )
