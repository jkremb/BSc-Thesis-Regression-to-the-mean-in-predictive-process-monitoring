import os

import pandas as pd
from matplotlib import pyplot as plt

results_postprocessing_folder = 'results_postprocessing'


def graph_arbConf(filepath, case):
    df = pd.read_csv(filepath, sep=",")

    df.plot(x='Confidence', y=['Total AUC before Rttm', 'Total AUC after Rttm'], kind='line', ylabel='Total AUC',
            title=case)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(results_postprocessing_folder, 'arbConf_' + case + '.jpg'))


sepsis_2_path = os.path.join(results_postprocessing_folder, 'arbConf_sepsis_2_graph_data.csv')
sepsis_4_path = os.path.join(results_postprocessing_folder, 'arbConf_sepsis_4_graph_data.csv')

graph_arbConf(sepsis_2_path, 'sepsis_2_logit_state_laststate')
graph_arbConf(sepsis_4_path, 'sepsis_4_logit_state_laststate')
