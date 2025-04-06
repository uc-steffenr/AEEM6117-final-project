"""Defines tools to visualize runs and metrics."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.embed_limit'] = 2**128


def find_nearest_ind(arr : np.ndarray, val : float) -> int:
    """Calculates index of value in array closes to the specified value.

    Parameters
    ----------
    arr : np.ndarray
        Array to be searched.
    val : float
        Value to be found.

    Returns
    -------
    int
        Index of nearest value in array.
    """
    return (np.abs(arr - val[:, None])).argmin(axis=1)


def plot_states(t : np.ndarray,
                y : np.ndarray,
                save : bool=True,
                show : bool=False,
                path : str=os.getcwd()) -> None:
    """Plots states of system.

    Parameters
    ----------
    t : np.ndarray
        Time steps.
    y : np.ndarray
        States.
    save : bool, optional
        Whether to save plot, by default True.
    show : bool, optional
        Whether to show plot, by default False.
    path : str, optional
        Path to save plot to, by default os.getcwd().
    """
    vars = ['theta_S', 'theta_1', 'theta_2', 'theta_T']
    fig, ax = plt.subplots(2, 2, sharex=True)
    ax = ax.ravel()

    [ax[i].plot(t, np.rad2deg(y[:, i])) for i in range(4)]
    [ax[i].grid() for i in range(4)]
    [ax[i].set_xlabel('time (s)') for i in [2, 3]]
    [ax[i].set_ylabel(f'$\{var}$ (deg)') for i, var in enumerate(vars)]

    fig.suptitle('System Angles')
    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(path, 'angles.jpg'))

    fig, ax = plt.subplots(2, 2, sharex=True)
    ax = ax.ravel()

    [ax[i].plot(t, np.rad2deg(y[:, i+4])) for i in range(4)]
    [ax[i].grid() for i in range(4)]
    [ax[i].set_xlabel('time (s)') for i in [2, 3]]
    [ax[i].set_ylabel(r'$\dot{'+f'\{var}'+'}$ (deg/s)') for i, var in enumerate(vars)]

    fig.suptitle('System Angular Velocities')
    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(path, 'angular_velocities.jpg'))


def plot_controls(t : np.ndarray,
                  u : np.ndarray,
                  save : bool=True,
                  show : bool=False,
                  path : str=os.getcwd()) -> None:
    """Plots controls of the system.

    Parameters
    ----------
    t : np.ndarray
        Time steps.
    u : np.ndarray
        Controls.
    save : bool, optional
        Whether to save plot, by default True.
    show : bool, optional
        Whether to show plot, by default False.
    path : str, optional
        Path to save plot to, by default os.getcwd().
    """
    fig, ax = plt.subplots(1, 1)

    ax.plot(t[:-1], u[:, 0], label=r'$\tau_S$')
    ax.plot(t[:-1], u[:, 1], label=r'$\tau_1$')
    ax.plot(t[:-1], u[:, 2], label=r'$\tau_2$')
    ax.set_ylabel('torque (Nm)')
    ax.set_xlabel('time (s)')
    ax.legend()
    ax.grid()

    fig.suptitle('System Controls')

    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(path, 'controls.jpg'))


def animate(y : np.ndarray,
            parameters : dict,
            path : str,
            **func_animate_kwargs) -> None:
    """Animates simulation.

    Parameters
    ----------
    y : np.ndarray
        States of system.
    parameters : dict
        Defining parameters of system.
    path : str
        Path to save animation to.
    """
    rho = parameters['rho']
    r_s = parameters['r_s']
    r_t = parameters['r_t']

    C = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    r_0 = lambda th_s: r_s + C(th_s)@np.array([rho[0], 0.])
    r_1 = lambda th_s, th_1: r_0(th_s) + C(th_s)@C(th_1)@np.array([2.*rho[1], 0.])
    r_2 = lambda th_s, th_1, th_2: r_1(th_s, th_1) + C(th_s)@C(th_1)@C(th_2)@np.array([2.*rho[2], 0.])
    r_c = lambda th: r_t + C(th)@np.array([0., rho[3]])

    fig, ax = plt.subplots()
    # TODO: come up with logic for setting xlim and ylim
    xlim = 1.5*np.max(np.abs([r_s[0], r_t[0]])) + np.max([rho[0], rho[3]]) + 2.*np.sum(rho[1:-1])
    ylim = 1.5*np.max(np.abs([r_s[1], r_t[1]])) + np.max([rho[0], rho[3]]) + 2.*np.sum(rho[1:-1])

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect('equal')

    main_sat_verts = np.array([
        [-rho[0], -rho[0]],
        [rho[0], -rho[0]],
        [rho[0], rho[0]],
        [-rho[0], rho[0]],
        [-rho[0], -rho[0]]
        ])

    targ_sat_verts = np.array([
        [-rho[3], -rho[3]],
        [rho[3], -rho[3]],
        [rho[3], rho[3]],
        [-rho[3], rho[3]],
        [-rho[3], -rho[3]]
    ])

    pt = r_c(y[0, 4])

    main_sat, = ax.plot([], [], 'b-', lw=2)
    p1 = ax.scatter(pt[0], pt[1], 55, 'black', '*', zorder=6)
    p2 = ax.scatter(pt[0], pt[1], 20, 'orange', '*', zorder=7)
    targ_sat, = ax.plot([], [], 'r-', lw=2)
    arm1, = ax.plot([], [], 'k.-', lw=2)
    arm2, = ax.plot([], [], 'k.-', lw=2)

    def init():
        r0 = r_0(y[0, 0])
        r1 = r_1(y[0, 0], y[0, 1])
        r2 = r_2(y[0, 0], y[0, 1], y[0, 2])
        rc = r_c(y[0, 4])

        main_verts = r_s[:, None] + C(y[0, 0])@main_sat_verts.T
        targ_verts = r_t[:, None] + C(y[0, 3])@targ_sat_verts.T

        main_sat.set_data(main_verts[0, :], main_verts[1, :])
        targ_sat.set_data(targ_verts[0, :], targ_verts[1, :])
        arm1.set_data([r0[0], r1[0]], [r0[1], r1[1]])
        arm2.set_data([r1[0], r2[0]], [r1[1], r2[1]])
        p1.set_offsets([rc[0], rc[1]])
        p2.set_offsets([rc[0], rc[1]])

        return main_sat, targ_sat, arm1, arm2, p1, p2

    def update(i):
        th_s = y[i, 0]
        th_1 = y[i, 1]
        th_2 = y[i, 2]
        th_t = y[i, 3]

        r0 = r_0(th_s)
        r1 = r_1(th_s, th_1)
        r2 = r_2(th_s, th_1, th_2)
        rc = r_c(th_t)

        main_verts = r_s[:, None] + C(th_s)@main_sat_verts.T
        targ_verts = r_t[:, None] + C(th_t)@targ_sat_verts.T

        main_sat.set_data(main_verts[0, :], main_verts[1, :])
        targ_sat.set_data(targ_verts[0, :], targ_verts[1, :])
        arm1.set_data([r0[0], r1[0]], [r0[1], r1[1]])
        arm2.set_data([r1[0], r2[0]], [r1[1], r2[1]])
        p1.set_offsets([rc[0], rc[1]])
        p2.set_offsets([rc[0], rc[1]])

        return main_sat, targ_sat, arm1, arm2, p1, p2

    if 'frames' not in func_animate_kwargs:
        func_animate_kwargs['frames'] = len(y)

    ani = FuncAnimation(fig, update, **func_animate_kwargs)
    ani.save(os.path.join(path, 'animation.gif'))
