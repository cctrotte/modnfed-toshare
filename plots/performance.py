import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
if __name__ == "__main__":

    load_path = 'cv_models_perf/'
    bsl_path = load_path + 'saved_bsl_bis/_hold_out_False.pkl'

    bsl_agg = load_path + 'saved_bsl_agg_bis/_hold_out_False.pkl'
    bsl_hold_out = load_path + 'saved_bsl_bis/_hold_out_True.pkl'
    bsl_sub_all =  load_path + 'saved_bsl_sub_bis/_alldata.pkl'

    bsl_03 =  load_path + 'saved_bsl_bis/_hold_out_False_03.pkl'
    bsl_05 =  load_path + 'saved_bsl_bis/_hold_out_False_05.pkl'
    bsl_07 =  load_path + 'saved_bsl_bis/_hold_out_False_07.pkl'

    bsl_holdout_03 =  load_path + 'saved_bsl_bis/_hold_out_True_03.pkl'
    bsl_holdout_05 =  load_path + 'saved_bsl_bis/_hold_out_True_05.pkl'
    bsl_holdout_07 =  load_path + 'saved_bsl_bis/_hold_out_True_07.pkl'

    bsl_alldata_03 = load_path + 'saved_bsl_sub_bis/_alldata_03.pkl'
    bsl_alldata_05 = load_path + 'saved_bsl_sub_bis/_alldata_05.pkl'
    bsl_alldata_07 = load_path + 'saved_bsl_sub_bis/_alldata_07.pkl'


    modn_path = load_path + 'saved_modn/_hold_out_False.pkl'
    modn_agg = load_path + 'saved_modn_agg/_hold_out_False.pkl'
    modn_hold_out = load_path + 'saved_modn/_hold_out_True.pkl'
    modn_sub_all = load_path + 'saved_modn_sub/_alldata.pkl'

    modn_03 =  load_path + 'saved_modn/_hold_out_False_03.pkl'
    modn_05 =  load_path + 'saved_modn/_hold_out_False_05.pkl'
    modn_07 =  load_path + 'saved_modn/_hold_out_False_07.pkl'

    modn_holdout_03 =  load_path + 'saved_modn/_hold_out_True_03.pkl'
    modn_holdout_05 =  load_path + 'saved_modn/_hold_out_True_05.pkl'
    modn_holdout_07 =  load_path + 'saved_modn/_hold_out_True_07.pkl'

    modn_alldata_03 =  load_path + 'saved_modn_sub/_alldata_03.pkl'
    modn_alldata_05 =  load_path + 'saved_modn_sub/_alldata_05.pkl'
    modn_alldata_07 =  load_path + 'saved_modn_sub/_alldata_07.pkl'


    name = 'all_data'
    bsl = pickle.load(open(bsl_path, 'rb'))
    modn = pickle.load(open(modn_path, 'rb'))

    bsl_agg = pickle.load(open(bsl_agg, 'rb'))
    modn_agg = pickle.load(open(modn_agg, 'rb'))

    bsl_hold_out = pickle.load(open(bsl_hold_out, 'rb'))
    modn_hold_out = pickle.load(open(modn_hold_out, 'rb'))

    bsl_sub_all = pickle.load(open(bsl_sub_all, 'rb'))
    modn_sub_all = pickle.load(open(modn_sub_all, 'rb'))

    # 30%, 50%, 70% missingness
    bsl_03 = pickle.load(open(bsl_03, 'rb'))
    bsl_05 = pickle.load(open(bsl_05, 'rb'))
    bsl_07 = pickle.load(open(bsl_07, 'rb'))

    modn_03 = pickle.load(open(modn_03, 'rb'))
    modn_05 = pickle.load(open(modn_05, 'rb'))
    modn_07 = pickle.load(open(modn_07, 'rb'))

    bsl_holdout_03 = pickle.load(open(bsl_holdout_03, 'rb'))
    bsl_holdout_05 = pickle.load(open(bsl_holdout_05, 'rb'))
    bsl_holdout_07 = pickle.load(open(bsl_holdout_07, 'rb'))

    modn_holdout_03 = pickle.load(open(modn_holdout_03, 'rb'))
    modn_holdout_05 = pickle.load(open(modn_holdout_05, 'rb'))
    modn_holdout_07 = pickle.load(open(modn_holdout_07, 'rb'))

    # model trained only on subset of data and evaluated on all data
    bsl_alldata_03 = pickle.load(open(bsl_alldata_03, 'rb'))
    bsl_alldata_05 = pickle.load(open(bsl_alldata_05, 'rb'))
    bsl_alldata_07 = pickle.load(open(bsl_alldata_07, 'rb'))

    modn_alldata_03 = pickle.load(open(modn_alldata_03, 'rb'))
    modn_alldata_05 = pickle.load(open(modn_alldata_05, 'rb'))
    modn_alldata_07 = pickle.load(open(modn_alldata_07, 'rb'))
    
    missing_dicts = {
        0.0: (bsl, modn),
        0.3: (bsl_03,   modn_03),
        0.5: (bsl_05,   modn_05),
        0.7: (bsl_07,   modn_07),
    }
    missing_dicts_holdout = {
        0.0: (bsl_hold_out, modn_hold_out),
        0.3: (bsl_holdout_03,   modn_holdout_03),
        0.5: (bsl_holdout_05,   modn_holdout_05),
        0.7: (bsl_holdout_07,   modn_holdout_07),
    }

    missing_dicts_alldata = {
        0.0: (bsl_sub_all, modn_sub_all),
        0.3: (bsl_alldata_03,   modn_alldata_03),
        0.5: (bsl_alldata_05,   modn_alldata_05),
        0.7: (bsl_alldata_07,   modn_alldata_07),
    }
    levels = list(missing_dicts.keys())
    labels = ['0%', '30%', '50%', '70%']

    # colors
    colors_ = {'modn': 'blue', 'mlp': 'grey', 'modn sub': 'cornflowerblue', 'mlp sub': 'lightgrey'}

    # Plotting
    # --- 1. Define which metrics and scenarios to plot
    metrics   = ['auprc', 'f1']
    types =  ['local', 'centralized', 'fl'] 

    # --- 2. Compute mean & std for each (scenario,metric)
    stats = {}
    for metric in metrics:
        stats[metric] = {
            'bsl': {
                'means': [np.mean(bsl[metric][sc]) for sc in types],
                'stds':  [np.std( bsl[metric][sc]) for sc in types],
            },
            'modn': {
                'means': [np.mean(modn[metric][sc]) for sc in types],
                'stds':  [np.std( modn[metric][sc]) for sc in types],
            }
        }

    x     = np.arange(len(types))
    width = 0.2

    color_map = {
    'Baseline':    'C0',
    'MoDN':        'C1',
    'Baseline-Agg':'C2',
    'MoDN-Agg':    'C3',
}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6,4))

        for i, sc in enumerate(types):
            # compute means/stds
            mb = np.mean(bsl[metric][sc]);  sb = np.std(bsl[metric][sc])
            mm = np.mean(modn[metric][sc]); sm = np.std(modn[metric][sc])

            # Baseline
            ax.bar(
                i - 1.5*width, mb, width,
                yerr=sb,
                color=color_map['Baseline'],
                label='Baseline'
            )
            # MoDN
            ax.bar(
                i - 0.5*width, mm, width,
                yerr=sm,
                color=color_map['MoDN'],
                label='MoDN'
            )

            # Only for FL, add the "agg" variants
            if sc == 'fl':
                mb_agg = np.mean(bsl_agg[metric]['fl'])
                sb_agg = np.std( bsl_agg[metric]['fl'])
                mm_agg = np.mean(modn_agg[metric]['fl'])
                sm_agg = np.std( modn_agg[metric]['fl'])

                ax.bar(
                    i + 0.5*width, mb_agg, width,
                    yerr=sb_agg,
                    color=color_map['Baseline-Agg'],
                    label='Baseline-Agg'
                )
                ax.bar(
                    i + 1.5*width, mm_agg, width,
                    yerr=sm_agg,
                    color=color_map['MoDN-Agg'],
                    label='MoDN-Agg'
                )

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in types])
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Scenario")
        ax.set_title(f"{metric.upper()} — Baseline vs MoDN (with FL-Agg)")

        # 2) dedupe legend so each label appears exactly once
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), ncol=2)

        plt.tight_layout()
        out = f"tmp/{name}_{metric}_with_agg.png"
        plt.savefig(out, dpi=300)
        plt.close(fig)

    # plot hold out also

    color_map = {
    'Baseline':    'C0',
    'MoDN':        'C1',
    'Baseline-HoldOut':'C2',
    'MoDN-HoldOut':    'C3',
    'Baseline-SubAll': 'C4',
    'MoDN-SubAll': 'C5'}
    width = 0.1
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6,4))

        for i, sc in enumerate(types):
            # compute means/stds
            mb = np.mean(bsl[metric][sc]);  sb = np.std(bsl[metric][sc])
            mm = np.mean(modn[metric][sc]); sm = np.std(modn[metric][sc])

            if sc != 'local':

                mb_h = np.mean(bsl_hold_out[metric][sc]);  sb_h = np.std(bsl_hold_out[metric][sc])
                mm_h = np.mean(modn_hold_out[metric][sc]); sm_h = np.std(modn_hold_out[metric][sc])

                mb_sub = np.mean(bsl_sub_all[metric][sc]);  sb_sub = np.std(bsl_sub_all[metric][sc])
                mm_sub = np.mean(modn_sub_all[metric][sc]); sm_sub = np.std(modn_sub_all[metric][sc])

            # Baseline
            ax.bar(
                i - 1.5*width, mb, width,
                yerr=sb,
                color=color_map['Baseline'],
                label='Baseline'
            )
            # MoDN
            ax.bar(
                i - 0.5*width, mm, width,
                yerr=sm,
                color=color_map['MoDN'],
                label='MoDN'
            )

            if sc != 'local':
                # Baseline hold out
                ax.bar(
                    i + 0.5 *width, mb_h, width,
                    yerr=sb_h,
                    color=color_map['Baseline-HoldOut'],
                    label='Baseline Hold Out'
                )
                # MoDN hold out
                ax.bar(
                    i + 1.5*width, mm_h, width,
                    yerr=sm_h,
                    color=color_map['MoDN-HoldOut'],
                    label='MoDN Hold Out'
                )
                ax.bar(
                    i + 2.5 *width, mb_sub, width,
                    yerr=sb_sub,
                    color=color_map['Baseline-SubAll'],
                    label='Baseline Sub All'
                )
                # MoDN hold out
                ax.bar(
                    i + 3.5*width, mm_sub, width,
                    yerr=sm_sub,
                    color=color_map['MoDN-SubAll'],
                    label='MoDN Sub All'
                )

        x_placement = [-.1, 1.1,2.1]
        ax.set_xticks(x_placement)
        ax.set_xticklabels([s.capitalize() for s in types])
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Scenario")
        ax.set_title(f"{metric.upper()} — Baseline vs MoDN (with HoldOut)")

        # 2) dedupe legend so each label appears exactly once
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), ncol=2, loc = "lower right")

        plt.tight_layout()
        out = f"tmp/{name}_{metric}w_holdout.png"
        plt.savefig(out, dpi=300)
        plt.close(fig)
    width = 0.1
    x = np.arange(len(levels))
    f= 7
    f_title = 10

    for metric in ['auprc']:
        for sc in ['fl']:
            means_bsl  = []
            stds_bsl   = []
            means_modn = []
            stds_modn  = []

            means_bsl_sub_alldata  = []
            stds_bsl_sub_alldata   = []
            means_modn_sub_alldata = []
            stds_modn_sub_alldata  = []

            means_bsl_holdout  = []
            stds_bsl_holdout = []
            means_modn_holdout = []
            stds_modn_holdout  = []

            means_modn_centralized =  []
            stds_modn_centralized = []

            means_modn_centralized_holdout =  []
            stds_modn_centralized_holdout = []

            means_modn_centralized_mixnmatch =  []
            stds_modn_centralized_mixnmatch = []

            means_modn_local = []
            stds_modn_local = []
            for lvl in levels:
                bsl_dict, modn_dict = missing_dicts[lvl]
                bsl_holdout_dict, modn_holdout_dict = missing_dicts_holdout[lvl]
                bsl_dict_sub_alldata, modn_dict_sub_alldata = missing_dicts_alldata[lvl]
                vals_bsl  = bsl_dict[metric][sc]
                vals_modn = modn_dict[metric][sc]

                vals_bsl_holdout = bsl_holdout_dict[metric][sc]
                vals_modn_holdout = modn_holdout_dict[metric][sc]

                vals_bsl_sub_alldata  = bsl_dict_sub_alldata[metric][sc]
                vals_modn_sub_alldata = modn_dict_sub_alldata[metric][sc]
                means_bsl.append(np.mean(vals_bsl))
                stds_bsl.append( np.std(vals_bsl))
                means_modn.append(np.mean(vals_modn))
                stds_modn.append(np.std(vals_modn))

                means_modn_holdout.append(np.mean(vals_modn_holdout))
                stds_modn_holdout.append(np.std(vals_modn_holdout))

                means_bsl_holdout.append(np.mean(vals_bsl_holdout))
                stds_bsl_holdout.append(np.std(vals_bsl_holdout))


                means_bsl_sub_alldata.append(np.mean(vals_bsl_sub_alldata))
                stds_bsl_sub_alldata.append(np.std(vals_bsl_sub_alldata))
                means_modn_sub_alldata.append(np.mean(vals_modn_sub_alldata))
                stds_modn_sub_alldata.append(np.std(vals_modn_sub_alldata))

                means_modn_centralized.append(np.mean(modn_dict[metric]['centralized']))
                stds_modn_centralized.append(np.std(modn_dict[metric]['centralized']))

                
                means_modn_centralized_holdout.append(np.mean(modn_holdout_dict[metric]['centralized']))
                stds_modn_centralized_holdout.append(np.std(modn_holdout_dict[metric]['centralized']))

                means_modn_centralized_mixnmatch.append(np.mean(modn_dict_sub_alldata[metric]['centralized']))
                stds_modn_centralized_mixnmatch.append(np.std(modn_dict_sub_alldata[metric]['centralized']))

                means_modn_local.append(np.mean(modn_dict[metric]['local']))
                stds_modn_local.append(np.std(modn_dict[metric]['local']))

            



            fig, ax = plt.subplots(figsize=(6,4))
            #   colors_ = {'modn': 'blue', 'mlp': 'grey', 'modn sub': 'cornflowerblue', 'mlp sub': 'lightgrey'}

            ax.bar(x-2*width, means_modn_local, width, yerr = stds_modn_local, label = 'MoDN (trained locally)', color = 'red')

            ax.bar(x - width, means_bsl,  width, yerr=stds_bsl,  label='FedMLP', color = colors_['mlp'])

            ax.bar(x , means_modn, width, yerr=stds_modn, label='FedMoDN', color = colors_['modn'])


            ax.bar(x +width, means_modn_centralized, width, yerr = stds_modn_centralized, label = 'MoDN (trained on centralized data)', color = 'green')


            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric.upper())
            ax.set_xlabel('Fraction of Missing Features')
            ax.set_title(f'FL training versus lower and upper baselines', fontsize = f_title)
            ax.legend(loc = 'lower right', fontsize = f)
            plt.tight_layout()

            out_dir = 'tmp/plots_missingness'
            os.makedirs(out_dir, exist_ok=True)
            fname = f'{out_dir}/{sc}_{metric}_missingness_1.png'
            plt.savefig(fname, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6,4))
            #   colors_ = {'modn': 'blue', 'mlp': 'grey', 'modn sub': 'cornflowerblue', 'mlp sub': 'lightgrey'}

            ax.bar(x - 2*width, means_bsl,  width, yerr=stds_bsl,  label='FedMLP', color = colors_['mlp'])
            ax.bar(x - width, means_bsl_sub_alldata,  width, yerr=stds_bsl_sub_alldata,  label='FedMLP MixNMatch', color= colors_['mlp sub'])

            ax.bar(x , means_modn, width, yerr=stds_modn, label='FedMoDN', color = colors_['modn'])
            ax.bar(x + width, means_modn_sub_alldata, width, yerr=stds_modn_sub_alldata, label='FedMoDN MixNMatch', color = colors_['modn sub'])

            ax.bar(x + 2*width, means_modn_centralized, width, yerr = stds_modn_centralized, label = 'MoDN centralized', color = 'green')
            ax.bar(x + 3*width, means_modn_centralized_mixnmatch, width, yerr = stds_modn_centralized_mixnmatch, label = 'MoDN centralized MixNMatch', color = 'darkgreen')


            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric.upper())
            ax.set_xlabel('Fraction of Missing Features')
            ax.set_title(f'Training on full features versus MixNMatch', fontsize =f_title)
            ax.legend(loc = 'lower right', fontsize = f)
            plt.tight_layout()

            out_dir = 'tmp/plots_missingness'
            os.makedirs(out_dir, exist_ok=True)
            fname = f'{out_dir}/{sc}_{metric}_missingness_2.png'
            plt.savefig(fname, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6,4))
            #   colors_ = {'modn': 'blue', 'mlp': 'grey', 'modn sub': 'cornflowerblue', 'mlp sub': 'lightgrey'}


            ax.bar(x - 2*width, means_bsl,  width, yerr=stds_bsl,  label='FedMLP', color = colors_['mlp'])
            ax.bar(x - width, means_bsl_holdout,  width, yerr=stds_bsl_holdout,  label='FedMLP (evaluate on holdout nodes)', color= colors_['mlp sub'])

            ax.bar(x, means_modn, width, yerr=stds_modn, label='FedMoDN', color = colors_['modn'])
            ax.bar(x + width, means_modn_holdout, width, yerr=stds_modn_holdout, label='FedMoDN (evaluate on holdout nodes)', color = colors_['modn sub'])

            ax.bar(x + 2*width, means_modn_centralized, width, yerr = stds_modn_centralized, label = 'MoDN centralized', color = 'green')
            ax.bar(x + 3*width, means_modn_centralized_holdout, width, yerr = stds_modn_centralized_holdout, label = 'MoDN centralized (evaluate on holdout nodes)', color = 'darkgreen')


            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric.upper())
            ax.set_xlabel('Fraction of Missing Features')
            ax.set_title(f'Evaluation on hold out test set versus evaluating on hold out nodes', fontsize = f_title)
            ax.legend(loc = 'lower right', fontsize = f)
            plt.tight_layout()

            out_dir = 'tmp/plots_missingness'
            os.makedirs(out_dir, exist_ok=True)
            fname = f'{out_dir}/{sc}_{metric}_missingness_3.png'
            plt.savefig(fname, dpi=300)
            plt.close(fig)
    
    # plot ensembling for modn
    width = 0.2
    x = np.arange(len(levels))

    for metric in metrics:
        for sc in types:
            means_bsl  = []
            stds_bsl   = []
            means_modn = []
            stds_modn  = []
            means_modn_ensemble = []
            stds_modn_ensemble = []
            for lvl in levels:
                bsl_dict, modn_dict = missing_dicts[lvl]
                vals_bsl  = bsl_dict[metric][sc]
                vals_modn = modn_dict[metric][sc]
                vals_modn_ensemble = modn_dict[metric + '_ensemble'][sc]
                means_bsl.append(np.mean(vals_bsl))
                stds_bsl.append( np.std(vals_bsl))
                means_modn.append(np.mean(vals_modn))
                stds_modn.append(np.std(vals_modn))
                means_modn_ensemble.append(np.mean(vals_modn_ensemble))
                stds_modn_ensemble.append(np.std(vals_modn_ensemble))


            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(x - width, means_bsl,  width, yerr=stds_bsl,  label='Baseline')
            ax.bar(x, means_modn, width, yerr=stds_modn, label='MoDN')
            ax.bar(x+width, means_modn_ensemble, width, yerr=stds_modn_ensemble, label = 'MoDN ensemble')

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric.upper())
            ax.set_xlabel('Fraction of Missing Features')
            ax.set_title(f'{sc.capitalize()} – {metric.upper()} vs Missingness')
            ax.legend()
            plt.tight_layout()

            out_dir = 'tmp/plots_missingness'
            os.makedirs(out_dir, exist_ok=True)
            fname = f'{out_dir}/{sc}_{metric}_missingness_w_ensemble.png'
            plt.savefig(fname, dpi=300)
            plt.close(fig)
    

    # custom plots

    # overall comparison
    modn_local = modn['auprc']['local']
    mlp_local = bsl['auprc']['local']

    mlp_fl = bsl['auprc']['fl']
    modn_fl = modn['auprc']['fl']

    modn_centralized = modn['auprc']['centralized']
    mlp_centralized = bsl['auprc']['centralized']

    fig, ax = plt.subplots(figsize=(6,4))
    width = 0.2
    x = np.arange(4)
    ax.bar(0, np.mean(modn_local), width, yerr = np.std(modn_local), label = 'MoDN Local', color = 'black')
    #ax.bar(width, np.mean(mlp_local), width, yerr = np.std(mlp_local), label = 'MLP Local')

    ax.bar(1, np.mean(mlp_fl), width, yerr = np.std(mlp_fl), label = 'MLP FL', color = colors_['mlp'])
    ax.bar(1 + width, np.mean(modn_fl), width, yerr = np.std(modn_fl), label = 'MoDN FL', color = colors_['modn'])

    ax.bar(2, np.mean(modn_centralized), width, yerr = np.std(modn_centralized), label = 'MoDN centralized', color = 'green')
    #ax.bar(3 + width, np.mean(mlp_centralized), width, yerr = np.std(mlp_centralized), label = 'MLP centralized')


    ax.legend(loc = 'lower right')
    plt.tight_layout()
    fname = f'tmp/modn_vs_baselines.png'
    plt.savefig(fname, dpi=300)


    # test data, hold out nodes, mix n match features
    mlp_fl_sub = bsl_sub_all['auprc']['fl']
    modn_fl_sub = modn_sub_all['auprc']['fl']

    mlp_fl_hold_out = bsl_hold_out['auprc']['fl']
    modn_fl_hold_out = modn_hold_out['auprc']['fl']

    fig, ax = plt.subplots(figsize = (6,4))
    width = 0.2

    x = [0.1, 1.1, 2.1]
    ax.bar(0, np.mean(mlp_fl), width, yerr = np.std(mlp_fl), color = colors_['mlp'])
    ax.bar(0 + width, np.mean(modn_fl), width, yerr = np.std(modn_fl),  color = colors_['modn'])

    ax.bar(1, np.mean(mlp_fl_hold_out), width, yerr = np.std(mlp_fl_hold_out),  color = colors_['mlp'])
    ax.bar(1 + width, np.mean(modn_fl_hold_out), width, yerr = np.std(modn_fl_hold_out),  color = colors_['modn'])

    ax.bar(2, np.mean(mlp_fl_sub), width, yerr = np.std(mlp_fl_sub), label = 'MLP', color = colors_['mlp'])
    ax.bar(2 + width, np.mean(modn_fl_sub), width, yerr = np.std(modn_fl_sub), label = 'MoDN', color = colors_['modn'])

    ax.legend(loc = 'lower right')
    ax.set_xticks(x)
    ax.set_xticklabels(['Hold out test data', 'Hold out nodes', 'MixnMatch training'])
    plt.tight_layout()
    fname = f'tmp/comparisons.png'
    plt.savefig(fname, dpi=300)

    print('End of script')