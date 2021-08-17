import matplotlib.pyplot as plt
from functools import partialmethod


class dynplot():
    """Extends matplotlib's pyplot.plot() to allow for repetitive plotting

        There is no simple way to update multiple lines repetitively and
        continuously of an existing figure in
        `matplotlib <https://matplotlib.org/>`_. Using this class as drop-in
        replacement for matplotlib's pyplot, the figure's line will be
        updated upon every call of the ``plot()`` method and create thus a
        dynamic plot, constantly refreshing.

        :ivar fig: ``matplotlib.figure.Figure`` instance of the figure
        :ivar ax: ``matplotlib.axes.Axes`` instance of the figure

        This modules contains a helper instance to simplify importing and use:

        >>> dplt = dynplot()

        Current limitations:

        - Only the ``plot`` function is supported.
        - The figure and axes are only configurable via the internal ``fig``
          and ``ax`` attribute, i.e. the following call will _fail_:

          >>> dplt.title('Will fail!')

          In this case, instead use the following:

          >>> _ = dplt.ax.set_title('Will work!')

        Example:

        >>> from dynplot import dynplot
        >>> from math import sin, pi
        >>>
        >>> dplt = dynplot()
        >>> for i in range(100):
        >>>     x = range(i, i+20)
        >>>     y = [sin(2*pi*x/20) for x in x]
        >>>     dplt.plot(x, y)
        >>>     _ = dplt.ax.set_title('Wave')
        >>>     dplt.show()

        :param refresh_rate: Refresh rate (in seconds), has a lower limit
                             given by the processing power of your machine
        :type refresh_rate: float
    """

    # TODO: Add and test more plotting functions
    supported_fcns = ['plot']

    def __init__(self, refresh_rate=0.1):
        # Configure object
        self.refresh_rate = refresh_rate

        # Create figure and axis
        self.fig, self.ax = plt.subplots()
        # self.fig, self.ax = plt.plot()

        # Set axis to auto-scale
        self.ax.set_autoscaley_on(True)

        self._initialized = False
        self._crnt_line = 0

    def _update(self, fcn, *args, **kwargs):
         # Create initial lines upon first calls
        if not self._initialized:
            lines = getattr(self.ax, fcn)(*args, **kwargs)

            # Verify if not consecutive call, adding multiple lines in
            # multiple steps (i.e. multiple calls of plot() before call to
            # show())
            if not hasattr(self, 'lines') or not self.lines:
                # create list if only one line existing
                if not isinstance(lines, list):
                    lines = [lines]

                self.lines = lines
            else:
                for line in lines:
                    self.lines.append(line)

        # Reuse existing lines upon following calls
        else:
            # Remove possible line styling indications from *args
            args = list(filter(lambda x: not isinstance(x, str), args))

            # Create set of lines to be updated
            if len(args) == 1:
                nbr_lines = 1
                single_line = True
            else:
                nbr_lines = len(args) // 2
                single_line = False

            # Only update parts of the lines
            if len(self.lines) > 1 and nbr_lines < len(self.lines):
                line_ids = list(range(self._crnt_line,
                                      self._crnt_line + nbr_lines))
                line_ids = [i % len(self.lines) for i in line_ids]

                self._crnt_line = (line_ids[-1] + 1) % len(self.lines)

            # Update all lines
            else:
                line_ids = list(range(0, len(self.lines)))
                self._crnt_line = 0

            # Apply changes to set of lines to be updated
            for i, line_id in enumerate(line_ids):
                # Set line values
                if single_line:
                    self.lines[line_id].set_ydata(args[i])
                else:
                    self.lines[line_id].set_xdata(args[2*i])
                    self.lines[line_id].set_ydata(args[2*i+1])

                # Set line attributes if existing
                for key, value in kwargs.items():
                    getattr(self.lines[line_id], 'set_' + key)(value)

    def __getattr__(self, name):
        if name in self.supported_fcns:

            def wrapper(*args, **kwargs):
                return self._update(name, *args, **kwargs)

            return wrapper

    def show(self, permanent=False, *args, **kwargs):
        """Displays figure

            Calls ``matplotlib.pyplot.pause()`` for continuous plotting or,
            if ``permanent`` is ``True`` forwards the call to
             ``matplotlib.pyplot.show()``

             :param permanent: Don't update or refresh plot
             :type permanent: bool
        """
        self._initialized = True

        # Rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # Draw and flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if permanent:
            plt.show(*args, **kwargs)
        else:
            plt.pause(self.refresh_rate)


# Helper Instance
dplt = dynplot()
