# from typing import List
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.lines import Line2D
# import numpy as np

# # from PySide6.QtCore import Signal, Slot


# class DataPlotter:
#     def __init__(self, axs: List[plt.Axes]):
#         # z_data_signal.connect(self.add_z_data)

#         self.time_history: List[float] = []

#         self.zref_history: List[float] = []
#         self.z_history: List[float] = []

#         self.theta_history: List[float] = []

#         self.force_history: List[float] = []

#         self.ax1 = axs[0]
#         self.ax2 = axs[1]
#         self.ax3 = axs[2]

#         self.z_fig = self.ax1.get_figure()
#         self.theta_fig = self.ax2.get_figure()
#         self.force_fig = self.ax3.get_figure()

#         self.z_plot = Plot(self.ax1, ylabel="z(m)", title="Ball on Beam Data")
#         self.theta_plot = Plot(self.ax2, ylabel="theta(deg)")
#         self.force_plot = Plot(self.ax3, xlabel="t(s)", ylabel="force(N)")

#     def update(self, t, reference, states, ctrl):
#         """
#         Add to the time and data histories, and update the plots.
#         """
#         print(f"\n\n ~~ GOT FORCE CTRL: {ctrl}")
#         # update the time history of all plot variables
#         with plt.ion():
#             self.time_history.append(t)
#             self.zref_history.append(reference)
#             self.z_history.append(states.item(0))
#             self.theta_history.append(
#                 180.0 / np.pi * states.item(1)
#             )  # rod angle (converted to degrees)
#             self.force_history.append(ctrl)

#             self.z_plot.update(self.time_history, [self.z_history, self.zref_history])
#             self.theta_plot.update(self.time_history, [self.theta_history])
#             self.force_plot.update(self.time_history, [self.force_history])

#             self.ax1.get_figure().canvas.draw()
#             self.ax2.get_figure().canvas.draw()
#             self.ax3.get_figure().canvas.draw()

#     # @Slot(float, float)
#     # def add_z_data(self, t: float, value: float):
#     #     self.ax1.plot(t, value, color="red", ls="-")
#     #     print(f"Plotting t,z = {t, value}")
#     #     self.z_fig.canvas.draw()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing
plt.style.use("dark_background")


class DataPlotter2:
    def __init__(self):
        self.time_history = []  # time
        self.zref_history = []  # reference position z_r
        self.z_history = []  # position z
        self.theta_history = []  # angle theta
        self.Force_history = []  # control force

        self.handle = []

    @property
    def z_ax(self):
        return self._z_ax

    @z_ax.setter
    def z_ax(self, value: plt.Axes):
        self._z_ax = value
        value.grid(color="#737373")
        self.handle.append(myPlot(self.z_ax, ylabel="z(m)", title="Part E States"))

    @property
    def theta_ax(self):
        return self._theta_ax

    @theta_ax.setter
    def theta_ax(self, value: plt.Axes):
        self._theta_ax = value
        value.grid(color="#737373")
        self.handle.append(myPlot(self.theta_ax, ylabel="theta(deg)"))

    @property
    def force_ax(self):
        return self._force_ax

    @force_ax.setter
    def force_ax(self, value: plt.Axes):
        self._force_ax = value
        value.grid(color="#737373")
        self.handle.append(myPlot(self.force_ax, xlabel="t(s)", ylabel="force(N)"))

    def update(self, t, reference, states, ctrl):
        """
        Add to the time and data histories, and update the plots.
        """
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        self.zref_history.append(reference)  # reference base position
        self.z_history.append(states.item(0))  # base position
        self.theta_history.append(
            180.0 / np.pi * states.item(1)
        )  # rod angle (converted to degrees)
        self.Force_history.append(ctrl)  # see controller return f,theta

        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.z_history, self.zref_history])
        self.handle[1].update(self.time_history, [self.theta_history])
        self.handle[2].update(self.time_history, [self.Force_history])

    def get_anim_ax(self):
        return self.ax[1, 1]


class Plot:
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
