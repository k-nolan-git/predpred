import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

# from pyrqa.time_series import TimeSeries
# from pyrqa.settings import Settings
# from pyrqa.analysis_type import Classic
# from pyrqa.neighbourhood import FixedRadius
# from pyrqa.metric import EuclideanMetric
# from pyrqa.computation import RQAComputation

folder = "C:/Users/B00955739/Documents/Git/phd/Init/"

sys.path.append(folder)

# import functions_v6_9 as fn
import ml_functions as mlf

# def add_metrics(sys_dict):
#     out_dict = sys_dict.copy()
#     out_dict['embed_dim'] = int(sys_dict['embed_dim'])

#     sim = np.load(sys_dict['sim_file'])
    
#     corrected_LDDP, corrected_LDDP_err = fn.calc_LDDP_corrected(sim)
    
#     rqa_ts = TimeSeries(sim, embedding_dimension=out_dict['embed_dim'], time_delay=1)
    
#     settings = Settings(rqa_ts)
#     computation = RQAComputation.create(settings, verbose=False)
#     result = computation.run()
#     rqa_rr = result.recurrence_rate
#     rqa_det = result.determinism
    
#     assert len(sim) == sys_dict['modelling_len'] + sys_dict['testing_len']
#     modelling_series = sim[:sys_dict['modelling_len']]
    
#     modelling_range = np.max(modelling_series) - np.min(modelling_series)
#     sim_range = np.max(sim) - np.min(sim)

#     out_dict['modelling_range'] = modelling_range
#     out_dict['sim_range'] = sim_range
#     out_dict['corrected_LDDP'] = corrected_LDDP
#     out_dict['corrected_LDDP_err'] = corrected_LDDP_err
#     out_dict['RQA_rr'] = rqa_rr
#     out_dict['RQA_det'] = rqa_det

#     return out_dict

def model_system(sys_dict):

    out_dict = sys_dict.copy()

    sim = np.load(sys_dict['sim_file'])
    
    assert len(sim) == sys_dict['modelling_len'] + sys_dict['testing_len']

    modelling_series = sim[:sys_dict['modelling_len']]
    testing_series = sim[-sys_dict['testing_len']:]
    
    assert len(modelling_series) == sys_dict['modelling_len']
    assert len(testing_series) == sys_dict['testing_len']

    test_dts = np.array(np.split(testing_series, sys_dict['n_tests']))
    assert np.all(test_dts.shape == (sys_dict['n_tests'], sys_dict['test_len']))

        
    random_seed = int(10*sys_dict['system_id'] + sys_dict['rep_id'])
    torch.manual_seed(random_seed)

    ##initialise model
    model = mlf.lstm(lstm_hs=sys_dict['layer_size'], lstm_nl=sys_dict['n_layers'])
    ## create dataloaders
    loader_dict = mlf.create_loaders(model_series=modelling_series, embed_dim=sys_dict['embed_dim'], train_split=0.8,
                                        test_series=testing_series, batch_size=sys_dict['batch_size'])
        
    trained_models, final_epoch, timings = mlf.train_model(model=model, train_loader=loader_dict["train_loader"],
                                    val_loader=loader_dict["val_loader"], epochs=sys_dict['epoch_vals'], patience=100,
                                    loss_fn=torch.nn.L1Loss(), opt=torch.optim.Adam, start_lr=sys_dict['lr'],
                                    lr_decay_factor=sys_dict['lr_decay_factor'], if_save=False, save_folder=None, save_name=None,
                                    plot=False, verbose=False)

    out_dict["final_epoch"] = final_epoch

    for epoch_idx in range(len(trained_models)):
        final_dict = out_dict.copy()
        trained_model = trained_models[epoch_idx]
        final_dict["epoch"] = sys_dict['epoch_vals']
        final_dict["model_time"] = timings[epoch_idx]

        final_dict["train_loss_mae"] = mlf.calc_loss(model=trained_model, dataloader=loader_dict["train_loader"], loss_fn=torch.nn.L1Loss())
        final_dict["val_loss_mae"] = mlf.calc_loss(model=trained_model, dataloader=loader_dict["val_loader"], loss_fn=torch.nn.L1Loss())
        final_dict.update(mlf.calc_point_metrics(model=trained_model, dataloader=loader_dict["test_loader"], prefix="test_loss"))
        
        pred_test_dts = mlf.pred_test_dts(test_dts=test_dts, model=trained_model, embed_dim=sys_dict['embed_dim'])

        for t in [0.001, 0.01, 0.05]:
            pred_h = calc_pred_h(test_dts=test_dts, pred_dts=pred_test_dts, embed_dim=sys_dict['embed_dim'], thresh=t,
                                    percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99], if_plot=False)
            final_dict.update(pred_h)

            if pred_h["pred_h_perfect_pred_count t={}".format(t)] > 0:
                test_dts_2 = np.array(np.split(testing_series, int(sys_dict['n_tests']/2)))
                pred_test_dts_2 = mlf.pred_test_dts(test_dts=test_dts_2, model=trained_model, embed_dim=sys_dict['embed_dim'])
                
                pred_h_2 = calc_pred_h(test_dts=test_dts_2, pred_dts=pred_test_dts_2, embed_dim=sys_dict['embed_dim'], thresh=t,
                                            percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99], if_plot=False)
                
                updated_pred_h_2 = {'{}_v2'.format(k): v for k, v in pred_h_2.items()}
                
                final_dict.update(updated_pred_h_2)
        
    return final_dict

def calc_pred_h(test_dts, pred_dts, embed_dim, thresh=0.01, percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99], if_plot=False):
    
    diff_all = np.abs(test_dts - pred_dts) ##compute abosolute error
    assert np.all(diff_all[:, 0:embed_dim] == 0) ## the first embed_dim values in each series are provided so should have no error
    diff = diff_all[:, embed_dim:] ## remove the provided values
    diff_acc = np.maximum.accumulate(diff, axis=1)

    if if_plot:
        avg_diff = np.mean(diff_acc, axis=0)
        max_diff = np.max(diff_acc, axis=0)
        min_diff = np.min(diff_acc, axis=0)
        lower_quart = np.percentile(diff_acc, 25, axis=0)
        upper_quart = np.percentile(diff_acc, 75, axis=0)
        iters = np.arange(1, len(avg_diff) + 1)
        
        plt.figure()
        plt.title("Error After N Iterations")
        plt.plot(iters, avg_diff, label="AVG")
        plt.plot(iters, max_diff, label="MAX")
        plt.plot(iters, min_diff, label="MIN")
        plt.plot(iters, lower_quart, label="25%")
        plt.plot(iters, upper_quart, label="75%")
        plt.hlines(y=thresh, xmin=0, xmax=1.05*max(iters), ls='--', color='black')
        plt.xlim(0, 1.05*max(iters))
        plt.ylabel("Absolute Error")
        plt.xlabel("No. Iterations")
        plt.legend()
        plt.show()
        plt.close()
        
    n_acc_iters = np.argmax(diff_acc > thresh, axis=1).astype(float)

    # if np.any(diff_acc[:, -1] <= thresh): ## if the error on the final predicted value for any series is below threshold:
    #     raise Exception("Entire series predicted accurately")
    
    perfect_pred_c = 0

    for i in range(len(n_acc_iters)):
        if diff_acc[i, -1] <= thresh:
            n_acc_iters[i] = np.inf
            perfect_pred_c += 1
        else:
            idx = int(n_acc_iters[i])
            assert diff_acc[i, idx] > thresh
            if idx > 0:
                assert diff_acc[i, idx-1] <= thresh

    result_dict = {}

    result_dict["pred_h_min t={}".format(thresh)] = np.min(n_acc_iters)
    result_dict["pred_h_max t={}".format(thresh)] = np.max(n_acc_iters)
    result_dict["pred_h_mean t={}".format(thresh)] = np.mean(n_acc_iters)
    percentile_arr = np.percentile(n_acc_iters, percentiles)
    percentile_arr[np.isnan(percentile_arr)] = np.inf
    for i in range(len(percentile_arr)):
        result_dict["pred_h_p{} t={}".format(percentiles[i], thresh)] = percentile_arr[i]
    result_dict["pred_h_sd t={}".format(thresh)] = np.std(n_acc_iters)
    result_dict["pred_h_perfect_pred_count t={}".format(thresh)] = perfect_pred_c
    
            
    if if_plot:
        n_acc_iters_noinf = n_acc_iters[np.isfinite(n_acc_iters)]
        if len(n_acc_iters_noinf) == 0:
            return result_dict

        plt.figure()
        plt.title("Prediction Horizon; t={}".format(thresh))
        hist = plt.hist(n_acc_iters_noinf, 
                        np.arange(-0.5, n_acc_iters_noinf.max() + 1), 
                        edgecolor = "black")
        plt.vlines(x=result_dict["pred_h_mean t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='red',
                    label="Mean of number of accurate iters: " + str(round(result_dict["pred_h_mean t={}".format(thresh)], 4)))
        plt.vlines(x=result_dict["pred_h_p50 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='green',
                    label="Median of number of accurate iters: " + str(round(result_dict["pred_h_p50 t={}".format(thresh)], 4)))
        plt.vlines(x=result_dict["pred_h_p25 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='black', ls='--',
                    label="Lower quartile")
        plt.vlines(x=result_dict["pred_h_p75 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='black', ls='--',
                    label="Upper quartile")
        plt.xlabel("Iterations Forward below " + str(thresh) + " error")
        plt.ylabel("Count")
        plt.ylim(0, 1.1*np.max(hist[0]))
        plt.xlim(n_acc_iters_noinf.min()-1, n_acc_iters_noinf.max()+1)
        plt.legend()
        plt.show()
        plt.close()

    return result_dict