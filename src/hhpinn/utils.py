import logging
import os

import matplotlib.pyplot as plt


def render_figure(fig=None, to_file='figure.pdf', save=False):
    """Save Matplotlib figure on disk if `save` is `True`, otherwise show it.

    If path `to_file` contains directories and they do not exists, then create
    the directories.

    Parameters
    ----------
    fig : handle (optional, default None)
        Handle to the figure to save. If `None`, then the current matplotlib
        figure is used.
    to_file : str
        Filename with or without directories. The extension of the filename
        determines the image format used.
    save : bool (optional, default False)
        Flag showing if the figure is saved to disk or plotted on screen.

    """
    if fig is None:
        fig = plt.gcf()

    if save:
        dirname = os.path.dirname(to_file)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        try:
            fig.savefig(to_file)
            logging.info("Save figure to file `%s`" % to_file)
        except IOError:
            logging.warning("Saving figure to file `%s` failed" % to_file)
    else:
        plt.show()
