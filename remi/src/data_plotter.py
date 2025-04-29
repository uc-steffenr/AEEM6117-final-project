import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

from typing import List, Dict


class DataPlotter:

    def __init__(self, axs: Dict[str, plt.Axes]):

        self.time_history: List[float] = []  # time
        self.thetas_history: List[float] = []  # satellite theta
        self.thetat_history: List[float] = []  # target theta
        self.theta1_history: List[float] = []  # theta 1
        self.theta2_history: List[float] = []  # theta 2

        self.taus_history: List[float] = []  # tau satellite
        self.tau1_history: List[float] = []  # tau 1
        self.tau2_history: List[float] = []  # tau 2
        self.eex_history: List[float] = []  # position of ee
        self.eey_history: List[float] = []  # position of ee
        self.targetx_history: List[float] = []  # ref position of target
        self.targety_history: List[float] = []  # ref position of target

        self.thetas_plot = SimplePlot(
            axs["thetas"], "Time (s)", "$\\theta_S$ (°)", "Satellite Angle"
        )
        self.thetat_plot = SimplePlot(
            axs["thetat"], "Time (s)", "$\\theta_T$ (°)", "Target Angle"
        )

        self.theta1_plot = SimplePlot(
            axs["theta1"], "Time (s)", "$\\theta_1$ (°)", "Joint Angle 1"
        )

        self.theta2_plot = SimplePlot(
            axs["theta2"], "Time (s)", "$\\theta_2$ (°)", "Joint Angle 2"
        )

        self.taus_plot = SimplePlot(
            axs["taus"], "Time (s)", "$\\tau_S$", "Satellite Torque"
        )

        self.tau1_plot = SimplePlot(
            axs["tau1"], "Time (s)", "$\\tau_1$", "Joint 1 Torque"
        )

        self.tau2_plot = SimplePlot(
            axs["tau2"], "Time (s)", "$\\tau_2$", "Joint 2 Torque"
        )

        self.ee_plot = SimplePlot(
            axs["ee"], "Time (s)", "EE", "EE Pose vs. Target Pose"
        )

    def update(self, t, states, ctrl):

        # def update(self, t, reference, states, ctrl):
        """

        Add to the time and data histories, and update the plots.
        """

        # update the time history of all plot variables

        thetaS = states.item(0)

        theta1 = states.item(1)

        theta2 = states.item(2)

        thetaT = states.item(3)

        tauS = ctrl.item(0)

        tau1 = ctrl.item(1)

        tau2 = ctrl.item(2)

        self.time_history.append(t)

        self.thetas_history.append(thetaS)

        self.theta1_history.append(theta1)

        self.theta2_history.append(theta2)

        self.thetat_history.append(thetaT)

        self.taus_history.append(tauS)

        self.tau1_history.append(tau1)

        self.tau2_history.append(tau2)

        with plt.ion():

            self.thetas_plot.update(self.time_history, [self.thetas_history])

            self.theta1_plot.update(self.time_history, [self.theta1_history])

            self.theta2_plot.update(self.time_history, [self.theta2_history])

            self.thetat_plot.update(self.time_history, [self.thetat_history])

            self.taus_plot.update(self.time_history, [self.taus_history])

            self.tau1_plot.update(self.time_history, [self.tau1_history])

            self.tau2_plot.update(self.time_history, [self.tau2_history])

            self.thetas_plot.ax.get_figure().canvas.draw()

            self.thetat_plot.ax.get_figure().canvas.draw()

            self.theta1_plot.ax.get_figure().canvas.draw()

            self.theta2_plot.ax.get_figure().canvas.draw()

            self.taus_plot.ax.get_figure().canvas.draw()

            self.tau1_plot.ax.get_figure().canvas.draw()

            self.tau2_plot.ax.get_figure().canvas.draw()

            self.ee_plot.ax.get_figure().canvas.draw()

    def get_anim_ax(self):

        return self.ax[1, 1]


class SimplePlot:
    """

    Create each individual subplot.
    """

    def __init__(self, ax, xlabel="", ylabel="", title="", legend=None):
        """

        ax - This is a handle to the  axes of the figure

        xlable - Label of the x-axis

        ylable - Label of the y-axis

        title - Plot title

        legend - A tuple of strings that identify the data.

                 EX: ("data1","data2", ... , "dataN")
        """

        self.legend = legend

        self.ax = ax  # Axes handle

        self.colors = ["b", "g", "r", "c", "m", "y", "b"]

        # A list of colors. The first color in the list corresponds

        # to the first line object, etc.

        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta

        # 'y' - yellow, 'k' - black

        self.line_styles = ["-", "-", "--", "-.", ":"]

        # A list of line styles.  The first line style in the list

        # corresponds to the first line object.

        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes

        self.ax.set_ylabel(ylabel)

        self.ax.set_xlabel(xlabel)

        self.ax.set_title(title)

        self.ax.grid(True)

        self.init = True

    def update(self, time, data):
        """

        Adds data to the plot.

        time is a list,

        data is a list of lists, each list corresponding to a line on the plot
        """

        if self.init == True:  # Initialize the plot the first time routine is called

            for i in range(len(data)):

                # Instantiate line object and add it to the axes
                self.line.append(
                    Line2D(
                        time,
                        data[i],
                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                        label=self.legend if self.legend != None else None,
                    )
                )

                self.ax.add_line(self.line[i])

            self.init = False

            # add legend if one is specified

            if self.legend != None:

                plt.legend(handles=self.line)

        else:  # Add new data to the plot

            # Updates the x and y data of each line.

            for i in range(len(self.line)):

                self.line[i].set_xdata(time)

                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data

        self.ax.relim()

        self.ax.autoscale()
