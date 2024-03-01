import matplotlib.pyplot as plt
import numpy as np

from functions import calc_metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def plot_metrics(models,X_tests,y_tests,predictions,df,nrows,ncols,index_y=10,samples=100):
    # models - array of models to plot from
    # X_test - array of test data for X
    # y_tests - array of test data fro y
    # df - The full dataframe with all columns and all datapoints

    cot_thin=3.6
    cot_med=23
    cot_thick=50

    #Get indexes for optical thin, med and thick
    test_indices=[]
    indices_thin=[]
    indices_med=[]
    indices_thick=[]
    for i,X_test in enumerate(X_tests):
        test_indices.append(X_test.index)
        df_tmp=df.iloc[test_indices[i]]

        #Reset index for future indexing
        df_tmp=df_tmp.reset_index()
        df_tmp=df_tmp.drop(columns=["index"])

        indices_thin.append(df_tmp[df_tmp['COT']<=cot_thin].index)
        indices_med.append(df_tmp[(df_tmp['COT']>cot_thin)&(df_tmp['COT']<=cot_med)].index)
        indices_thick.append(df_tmp[df_tmp['COT']>cot_med].index)


    figs=[]
    axs=[]


    fig_1, ax_1=plt.subplots(nrows=nrows,ncols=ncols)
    fig_1.suptitle("Training/Validation loss")

    for i,ax in enumerate(ax_1.ravel()):
        ax.plot(models[i].train_loss.data.cpu().numpy())
        ax.plot(models[i].val_loss.data.cpu().numpy())
        ax.set_title("All channels estimated")
        ax.legend(['Training Loss','Validation Loss'])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

    figs.append(fig_1)
    axs.append(ax_1)

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

        residuals.append(y_test_tmp-y_pred_tmp[:,1])
        y_true.append(y_test_tmp)
        y_pred.append(y_pred_tmp)

    ## Plot residuals ##
    fig_2,ax_2=plt.subplots(nrows=nrows,ncols=ncols)
    fig_2.suptitle("Residual plots.")

    for i,ax in enumerate(ax_2.ravel()):
        ax.plot(y_true[i][indices_thin[i]],residuals[i][indices_thin[i]],'.', markersize=2)
        ax.plot(y_true[i][indices_med[i]],residuals[i][indices_med[i]],'.', markersize=2)
        ax.plot(y_true[i][indices_thick[i]],residuals[i][indices_thick[i]],'.', markersize=2)
        ax.set_title("Model "+str(i))
        ax.hlines(0,xmin=-1,xmax=10,colors='r')
        ax.set_xlim((np.min(y_true[i])-0.2,np.max(y_true[i])+0.2))
        ax.legend(['Residual thin COT <3.6',
                        'Residual med COT <23',
                        'Residual thick COT <50',
                        'Zero error line'])
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Residual")


    figs.append(fig_2)
    axs.append(ax_2)

    ## Plot prediction v ground truth ##
    fig_3,ax_3=plt.subplots(nrows=nrows,ncols=ncols)
    fig_3.suptitle("Prediction vs Ground Truth")
    for i,ax in enumerate(ax_3.ravel()):
        
        ax.plot(y_pred[i][indices_thin[i],1],y_true[i][indices_thin[i]],'.', markersize=1)
        ax.plot(y_pred[i][indices_med[i],1],y_true[i][indices_med[i]],'.', markersize=1)
        ax.plot(y_pred[i][indices_thick[i],1],y_true[i][indices_thick[i]],'.', markersize=1)
        #plt.plot(cloudy_sort,'.')
        line=np.linspace(0,1,100)
        ax.plot(line,line)
        ax.legend(['Predictions thin COT <3.6',
                        'Predictions med COT <23',
                        'Predictions thick COT <50',
                        'Zero Error line'])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

    figs.append(fig_3)
    axs.append(ax_3)
        
    ## Prediction v Ground Truth ##
    bins=np.linspace(0,1,100)
    
    fig_4,ax_4=plt.subplots(nrows=nrows,ncols=ncols)
    fig_4.suptitle("Prediction vs Ground Truth, averaged")
    for i,ax in enumerate(ax_4.ravel()):
        freq_true=np.zeros(len(bins))
        freq_pred=np.zeros(len(bins))
        for j,edge in enumerate(bins):
            if j!=0:
                indices=(y_true[i]>bins[j-1])&(y_true[i]<=edge)
                mean_bin_true=np.mean(y_true[i][indices])
                mean_bin_pred=np.mean(y_pred[i][indices])
                
                freq_true[j]=mean_bin_true
                freq_pred[j]=mean_bin_pred
        ax.plot(freq_pred,freq_true,'.')
        #plt.plot(cloudy_sort,'.')
        line=np.linspace(0,1,100)
        ax.plot(line,line)
        ax.legend(['Predictions'])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

    figs.append(fig_4)
    axs.append(ax_4)

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
    fig_5, ax_5 = plt.subplots(nrows=nrows,ncols=ncols)
    fig_5.suptitle("Reflectivity and uncertainty")
    for i,ax in enumerate(ax_5.ravel()):
        ax.plot(y_true_sort_samp[i],'.',label='Ground Truth')
        ax.errorbar(x=np.linspace(0,len(y_pred_sort_samp[i][:,1]),len(y_pred_sort_samp[i][:,1]))
                    ,y=y_pred_sort_samp[i][:,1],
                    yerr=[np.abs(y_pred_sort_samp[i][:,1]-y_pred_sort_samp[i][:,0]),
                        np.abs(y_pred_sort_samp[i][:,1]-y_pred_sort_samp[i][:,2])],
                            marker='.',fmt='.',label='Predictions')

        percent=0.1
        ax.plot(y_true_sort_samp[i]-y_true_sort_samp[i]*percent,'g',label='percent')
        ax.plot(y_true_sort_samp[i]+y_true_sort_samp[i]*percent,'g',label='percent')
        ax.set_xlabel("Arbitrary samples")
        ax.set_ylabel("Reflectivity")
        ax.legend()

    figs.append(fig_5)
    axs.append(ax_5)


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



    return figs, axs, MSE, R2, PSNR, quantrates, quantcross