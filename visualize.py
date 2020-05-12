import torch
import numpy as np
from visdom import Visdom

viz = Visdom()


def scatter(points, win='data', name='data', color=[0,0,0]):
    viz.scatter(
        X=points,
        win=win,
        name=name,
        opts=dict(
            title=win,
            markersize=4,
            markerborderwidth=0,
            markercolor=np.array([color]),
            xtickmin=-30,
            xtickmax=30,
            ytickmin=-30,
            ytickmax=30
        )
    )


line_data = {}
def line(x, y, win):
    if win not in line_data: line_data[win] = {'x': [], 'y': []}
    line_data[win]['x'].append(x)
    line_data[win]['y'].append(y)
    viz.line(
        X=np.array(line_data[win]['x']),
        Y=np.array(line_data[win]['y']),
        win=win,
        opts=dict(
            title=win
        )
    )


def heatmap(points, win, x_labels, y_labels):
    viz.heatmap(
        X=points.numpy().transpose(),
        win=win,
        opts=dict(
            title=win,
            columnnames=x_labels,
            rownames=y_labels
        )
    )


def map(fn, name):
    x_range = torch.arange(-30, 30)
    y_range = torch.arange(-30, 30)
    arr = torch.zeros((len(x_range), len(y_range)))
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            x, y = x_range[i], y_range[j]
            arr[i][j] = fn(torch.FloatTensor([x, y]))
    heatmap(arr, name, x_range.tolist(), y_range.tolist())