import numpy as np
import torch
import sys

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

folder = "C:/Users/B00955739/Documents/Git/phd/Init/"

sys.path.append(folder)

import functions_v6_8 as fn
import ml_functions as mlf

def add_metrics(sys_dict):
    out_dict = sys_dict.copy()
    out_dict['embed_dim'] = int(sys_dict['embed_dim'])

    sim = np.load(sys_dict['sim_file'])
    
    corrected_LDDP, corrected_LDDP_err = fn.calc_LDDP_corrected(sim)
    
    rqa_ts = TimeSeries(sim, embedding_dimension=out_dict['embed_dim'], time_delay=1)
    
    settings = Settings(rqa_ts)
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    rqa_rr = result.recurrence_rate
    rqa_det = result.determinism
    
    assert len(sim) == sys_dict['modelling_len'] + sys_dict['testing_len']
    modelling_series = sim[:sys_dict['modelling_len']]
    
    modelling_range = np.max(modelling_series) - np.min(modelling_series)
    sim_range = np.max(sim) - np.min(sim)

    out_dict['modelling_range'] = modelling_range
    out_dict['sim_range'] = sim_range
    out_dict['corrected_LDDP'] = corrected_LDDP
    out_dict['corrected_LDDP_err'] = corrected_LDDP_err
    out_dict['RQA_rr'] = rqa_rr
    out_dict['RQA_det'] = rqa_det

    return out_dict

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
            pred_h = fn.calc_pred_h(test_dts=test_dts, pred_dts=pred_test_dts, embed_dim=sys_dict['embed_dim'], thresh=t,
                                    percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99], if_plot=False)
            final_dict.update(pred_h)
        
    return final_dict