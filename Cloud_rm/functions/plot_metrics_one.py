import matplotlib.pyplot as plt
import numpy as np

from functions import calc_metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def plot_metrics_one(models,X_tests,y_tests,predictions,df,index_y=10,samples=100,is_ensemble=True,index_median=1):
    # models - array of models to plot from
    # X_test - array of test data for X
    # y_tests - array of test data fro y
    # df - The full dataframe with all columns and all datapoints
    n_plots=4

    cot_thin=3.6
    cot_med=23
    cot_thick=50

    fig,axs=plt.subplots(ncols=1,nrows=n_plots)
    f_i=0 #Figure index

    #Get indexes for optical thin, med and thick
    test_indices=[]
    indices_zero=[]
    indices_thin=[]
    indices_med=[]
    indices_thick=[]
    for i,X_test in enumerate(X_tests):
        test_indices.append(X_test.index)
        df_tmp=df.iloc[test_indices[i]]

        #Reset index for future indexing
        df_tmp=df_tmp.reset_index()
        df_tmp=df_tmp.drop(columns=["index"])

        indices_zero.append(df_tmp[df_tmp['COT']==0].index)
        indices_thin.append(df_tmp[(df_tmp['COT']<=cot_thin)&df_tmp['COT']>0].index)
        indices_med.append(df_tmp[(df_tmp['COT']>cot_thin)&(df_tmp['COT']<=cot_med)].index)
        indices_thick.append(df_tmp[df_tmp['COT']>cot_med].index)




    ## Calculate residuals ##
    residuals=[]
    y_true=[]
    y_pred=[]
    for i, model in enumerate(models):
        y_test_tmp=y_tests[i].to_numpy()
        y_pred_tmp=predictions[i]
        if len(y_test_tmp[0,:])>1:
            y_test_tmp=y_test_tmp[:,index_y]
            y_pred_tmp=y_pred_tmp[:,index_y]
        else:
            y_test_tmp=y_test_tmp[:,0]

        residuals.append(y_test_tmp-y_pred_tmp[:,index_median])
        y_true.append(y_test_tmp)
        y_pred.append(y_pred_tmp)

    ## Plot residuals ##



    axs[f_i].plot(y_true[i][indices_thin[i]],residuals[i][indices_thin[i]],'.', markersize=2)
    axs[f_i].plot(y_true[i][indices_med[i]],residuals[i][indices_med[i]],'.', markersize=2)
    axs[f_i].plot(y_true[i][indices_thick[i]],residuals[i][indices_thick[i]],'.', markersize=2)
    axs[f_i].set_title("Model "+str(i))
    axs[f_i].hlines(0,xmin=-1,xmax=10,colors='r')
    axs[f_i].set_xlim((np.min(y_true[i])-0.2,1.2))
    axs[f_i].legend(['Residual thin COT <3.6',
                    'Residual med COT <23',
                    'Residual thick COT <50',
                    'Zero error line'])
    axs[f_i].set_xlabel("Ground Truth")
    axs[f_i].set_ylabel("Residual")

    ##
    f_i+=1
    ##
    
    ## Plot prediction v ground truth ##

    axs[f_i].plot(y_pred[i][indices_thin[i],1],y_true[i][indices_thin[i]],'.', markersize=1)
    axs[f_i].plot(y_pred[i][indices_med[i],1],y_true[i][indices_med[i]],'.', markersize=1)
    axs[f_i].plot(y_pred[i][indices_thick[i],1],y_true[i][indices_thick[i]],'.', markersize=1)
    #plt.plot(cloudy_sort,'.')
    line=np.linspace(0,1,100)
    axs[f_i].plot(line,line)
    axs[f_i].set_xlim((np.min(y_pred[i])-0.2,1.2))
    axs[f_i].set_ylim((np.min(y_true[i])-0.2,1.2))
    axs[f_i].legend(['Predictions thin COT <3.6',
                    'Predictions med COT <23',
                    'Predictions thick COT <50',
                    'Zero Error line'])
    axs[f_i].set_xlabel("Prediction")
    axs[f_i].set_ylabel("Ground Truth")
    
    ##
    f_i+=1
    ##
        
    ## Prediction v Ground Truth ##
    bins=np.linspace(0,1,100)
    
    freq_true=np.zeros(len(bins))
    freq_pred=np.zeros(len(bins))

    freq_true_zero=np.zeros(len(bins))
    freq_pred_zero=np.zeros(len(bins))

    freq_true_thin=np.zeros(len(bins))
    freq_pred_thin=np.zeros(len(bins))

    freq_true_med=np.zeros(len(bins))
    freq_pred_med=np.zeros(len(bins))

    y_tmp_zero=y_true[i][indices_zero[i]]
    y_tmp_thin=y_true[i][indices_thin[i]]
    y_tmp_med=y_true[i][indices_med[i]]

    y_tmp_pred_zero=y_pred[i][indices_zero[i]]
    y_tmp_pred_thin=y_pred[i][indices_thin[i]]
    y_tmp_pred_med=y_pred[i][indices_med[i]]

    for j,edge in enumerate(bins):
        if j!=0:
            indices=(y_true[i]>bins[j-1])&(y_true[i]<=edge)

            ind_z=(y_tmp_zero>bins[j-1])&(y_tmp_zero<=edge)
            ind_t=(y_tmp_thin>bins[j-1])&(y_tmp_thin<=edge)
            ind_m=(y_tmp_med>bins[j-1])&(y_tmp_med<=edge)

            mean_bin_true=np.mean(y_true[i][indices])
            mean_bin_pred=np.mean(y_pred[i][indices])

            mean_bin_true_zero=np.mean(y_tmp_zero[ind_z])
            mean_bin_pred_zero=np.mean(y_tmp_pred_zero[ind_z])

            mean_bin_true_thin=np.mean(y_tmp_thin[ind_t])
            mean_bin_pred_thin=np.mean(y_tmp_pred_thin[ind_t])

            mean_bin_true_med=np.mean(y_tmp_med[ind_m])
            mean_bin_pred_med=np.mean(y_tmp_pred_med[ind_m])
            
            freq_true[j]=mean_bin_true
            freq_pred[j]=mean_bin_pred

            freq_true_zero[j]=mean_bin_true_zero
            freq_pred_zero[j]=mean_bin_pred_zero

            freq_true_thin[j]=mean_bin_true_thin
            freq_pred_thin[j]=mean_bin_pred_thin

            freq_true_med[j]=mean_bin_true_med
            freq_pred_med[j]=mean_bin_pred_med

    axs[f_i].plot(freq_pred,freq_true,'.')
    axs[f_i].plot(freq_pred_zero,freq_true_zero,'.')
    axs[f_i].plot(freq_pred_thin,freq_true_thin,'.')
    axs[f_i].plot(freq_pred_med,freq_true_med,'.')
    #plt.plot(cloudy_sort,'.')
    line=np.linspace(0,1,100)
    axs[f_i].plot(line,line)
    axs[f_i].legend(['All predictions','Cloud free','Thin predictions','Medium predictions'])
    axs[f_i].set_xlabel("Prediction")
    axs[f_i].set_ylabel("Ground Truth")

    ##
    f_i+=1
    ##

    ## Uncertainties and percentages ##
    # Sort values #

    y_true_sort=[]
    y_pred_sort=[]
    for i, y in enumerate(y_true):
        sort=np.argsort(y)
        y_true_sort.append(y[sort])
        y_pred_sort.append(y_pred[i][sort])

    y_true_sort_samp=[]
    y_pred_sort_samp=[]
    for i, y in enumerate(y_true):
        y_samp=y[:samples]
        sort=np.argsort(y_samp)
        y_true_sort_samp.append(y_samp[sort])
        y_pred_sort_samp.append(y_pred[i][:samples][sort])

    # Plot the values
    axs[f_i].plot(y_true_sort_samp[i],'.',label='Ground Truth')
    axs[f_i].errorbar(x=np.linspace(0,len(y_pred_sort_samp[i][:,index_median]),len(y_pred_sort_samp[i][:,index_median]))
                ,y=y_pred_sort_samp[i][:,index_median],
                yerr=[np.abs(y_pred_sort_samp[i][:,index_median]-y_pred_sort_samp[i][:,0]),
                    np.abs(y_pred_sort_samp[i][:,index_median]-y_pred_sort_samp[i][:,-1])],
                        marker='.',fmt='.',label='Predictions')

    percent=0.1
    axs[f_i].plot(y_true_sort_samp[i]-y_true_sort_samp[i]*percent,'g',label='percent')
    axs[f_i].plot(y_true_sort_samp[i]+y_true_sort_samp[i]*percent,'g',label='percent')
    axs[f_i].set_xlabel("Arbitrary samples")
    axs[f_i].set_ylabel("Reflectivity")
    axs[f_i].set_ylim((0,1.2))
    axs[f_i].legend()

    


    ## Calculate some metrics ##
    MSE=[]
    R2=[]
    PSNR=[]
    quantrates=[]
    quantcross=[]
    for i, model in enumerate(models):
        MSE.append(mean_squared_error(y_true[i],y_pred[i][:,1]))
        R2.append(r2_score(y_true[i],y_pred[i][:,1]))
        PSNR.append(calc_metrics.PSNR(y_true[i],y_pred[i][:,1]))
        quantrates.append(calc_metrics.quant_rate(y_true[i],y_pred[i]))
        quantcross.append(calc_metrics.quant_cross(y_pred[i]))



    return fig, axs, MSE, R2, PSNR, quantrates, quantcross