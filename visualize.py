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