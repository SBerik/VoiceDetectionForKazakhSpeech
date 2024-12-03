import plotly.graph_objects as go

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