"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from matplotlib.patches import Rectangle

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def getDecisionErrorData():
    data = [
        ['DE1', 'DE2', 'DE3', 'DE4', 'DE5', 'DE6', 'DE7', 'DE8'],
        ('Decision Error', [
            [4.93, 2.87, 6.60, 0.56, 1.88, 0.09, 0.22, 9.09],
            [24.63, 14.37, 13.20, 11.14, 9.38, 4.40, 4.40, 18.18]])
    ]
    return data

def getPerformanceErrorData():
    data = [
        ['PE1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6', 'PE7'],
        ('Non/Performance Error', [
            [5.4, 10.32, 2.5, 3.64, 1.49, 1.1, 4.4],
            [26.92, 25.82, 17.58, 13.19, 1.65, 2.20, 8.79]])
    ]
    return data

def getRecognitionErrorData():
    data = [
        ['RE1', 'RE2', 'RE3', 'RE4', 'RE5'],
        ('Recognition Error', [
            [10, 10.54, 6.74, 5.16, 3.08],
            [50, 26.35, 9.36, 7.88, 6.16]])
    ]
    return data

#if __name__ == '__main__':

def plotDecisionErrors():
    N = 8
    theta = radar_factory(N, frame='polygon')

    data = getDecisionErrorData()
    spoke_labels = data.pop(0)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.text(0.1, 0.425, 'DE1: Too Fast for Conditions')
    fig.text(0.1, 0.40, 'DE2: Too Fast for Curve')
    fig.text(0.1, 0.375, 'DE3: False Assumption of Other\'s Action')
    fig.text(0.1, 0.35, 'DE4: Illegal Maneuver')
    fig.text(0.1, 0.325, 'DE5: Misjudgment of Gap or Other\'s Speed')
    fig.text(0.1, 0.30, 'DE6: Following Too Closely')
    fig.text(0.1, 0.275, 'DE7: Aggressive Driving Behavior')
    fig.text(0.1, 0.25, 'DE8: Unknown Decision Error')
    
    # colors = ['b', 'r', 'g', 'm', 'y']
    colors = ['b', 'r']
    # Plot the four cases from the example data on separate axes
    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([5, 10, 15, 20, 25, 30])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
##    plt.subplot(2, 2, 1)
##    labels = ('Conventional Vehicles', 'Vehicles with Cognition')
##    legend = plt.legend(labels, loc=(-0.1, -0.3), labelspacing=0.1)
##    plt.setp(legend.get_texts(), fontsize='small')
    
    # plt.figtext(0.5, 0.965, 'Comparison of Critical Reasons for Pre-Crash Events',
    #            ha='center', color='black', weight='bold', size='large')
    plt.show()

def plotPerformanceErrors():
    N = 7
    theta = radar_factory(N, frame='polygon')

    data = getPerformanceErrorData()
    spoke_labels = data.pop(0)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.text(0.075, 0.425, 'PE1: Overcompensation')
    fig.text(0.075, 0.40, 'PE2: Poor Directional Control')
    fig.text(0.075, 0.375, 'PE3: Sleep')
    fig.text(0.075, 0.35, 'PE4: Heart Attack or Other Physical Impairment')
    fig.text(0.075, 0.325, 'PE5: Panic/Freezing')
    fig.text(0.075, 0.30, 'PE6: Unknown Performance Error')
    fig.text(0.075, 0.275, 'PE7: Unknown Critical Nonperformance Error')
    
    colors = ['b', 'r']
    # Plot the four cases from the example data on separate axes
    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([5, 10, 15, 20, 25, 30])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    plt.show()

def plotRecognitionErrors():
    N = 5
    theta = radar_factory(N, frame='polygon')

    data = getRecognitionErrorData()
    spoke_labels = data.pop(0)

    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.text(0.15, 0.425, 'RE1: Inadequate Surveillance')
    fig.text(0.15, 0.40, 'RE2: Internal Distraction')
    fig.text(0.15, 0.375, 'RE3: External Distraction')
    fig.text(0.15, 0.35, 'RE4: Inattention')
    fig.text(0.15, 0.325, 'RE5: Unknown Recognition Error')
    
    colors = ['b', 'r']
    # Plot the four cases from the example data on separate axes
    for n, (title, case_data) in enumerate(data):
        ax = fig.add_subplot(2, 2, n + 1, projection='radar')
        plt.rgrids([10, 20, 30, 40, 50])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    plt.show()
