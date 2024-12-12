import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt


def plotly_plot_roc (fprs, tprs, auc_value, desire_fr_val_i = None, desire_tr_val_i = None, vertical_line = False):
    trace = go.Scatter(x=fprs, y=tprs, mode='lines', name='AUC = %0.2f' % auc_value,
                       line=dict(color='darkorange', width=2))
    reference_line = go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Reference Line',
                                line=dict(color='navy', width=2, dash='dash'))

    if desire_fr_val_i is not None and desire_tr_val_i is not None:
        selected_point = go.Scatter(x=[desire_fr_val_i], 
                                     y=[desire_tr_val_i], 
                                     mode='markers', 
                                     name='Selected Point',
                                     marker=dict(color='red', size=12))
        if vertical_line:
            vertical_line = go.Scatter(x=[desire_fr_val_i, desire_fr_val_i], 
                                        y=[0, desire_tr_val_i], 
                                        mode='lines',
                                        showlegend=False,
                                        line=dict(color='rgb(255, 82, 82)', width=1, dash='dot'))
            fig = go.Figure(data=[trace, reference_line, vertical_line, selected_point])
        else:
            fig = go.Figure(data=[trace, reference_line, selected_point])  
    else:
        fig = go.Figure(data=[trace, reference_line])
    
    fig.update_layout(xaxis_title='FPR',
                      yaxis_title='TPR',
                      height=700)
    return fig


def save_graph_tb_log_metrics(first_csv_path, second_csv_path, name_ox, name_oy, pth_save = '../pics/metrics_plot.png'):
    """
    Builds a plot of metric dependencies from two CSV files and saves the image.
    Parameters:
    - first_csv_path (str): Path to the first CSV file (Train).
    - second_csv_path (str): Path to the second CSV file (Validation).
    - name_ox (str): Name for the X axis.
    - name_oy (str): Name for the Y axis.
    - plot (bool): If True, shows the plot.
    - saving (bool): If True, saves the plot to a file named `metrics_plot.png`.
    """
    m_train, m_val  = pd.read_csv(first_csv_path), pd.read_csv(second_csv_path)
    epochs_train, values_train = m_train['Step'], m_train['Value']
    epochs_val, values_val = m_val['Step'], m_val['Value']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, values_train, marker='o', linestyle='--', color='red', label=f"Train {name_oy}")
    plt.plot(epochs_val, values_val, marker='o', linestyle='-', color='blue', label=f"Validation {name_oy}")
    plt.xlabel(name_ox)
    plt.ylabel(name_oy)
    plt.legend(loc='lower right') 
    plt.grid(True)
    plt.savefig(pth_save, dpi=300)
    print(f"График сохранён в {pth_save}")