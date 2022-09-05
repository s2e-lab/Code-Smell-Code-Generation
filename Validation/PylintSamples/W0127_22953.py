def set_rcParams_scvelo(fontsize=8, color_map=None, frameon=None):
    """Set matplotlib.rcParams to scvelo defaults."""

    # dpi options (mpl default: 100, 100)
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 150

    # figure (mpl default: 0.125, 0.96, 0.15, 0.91)
    rcParams['figure.figsize'] = (7, 5)
    rcParams['figure.subplot.left'] = 0.18
    rcParams['figure.subplot.right'] = 0.96
    rcParams['figure.subplot.bottom'] = 0.15
    rcParams['figure.subplot.top'] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams['lines.linewidth'] = 1.5  # the line width of the frame
    rcParams['lines.markersize'] = 6
    rcParams['lines.markeredgewidth'] = 1

    # font
    rcParams['font.sans-serif'] = \
        ['Arial', 'Helvetica', 'DejaVu Sans',
         'Bitstream Vera Sans', 'sans-serif']

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    # fonsizes (mpl default: 10, medium, large, medium)
    rcParams['font.size'] = fontsize
    rcParams['legend.fontsize'] = labelsize
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = labelsize

    # legend (mpl default: 1, 1, 2, 0.8)
    rcParams['legend.numpoints'] = 1
    rcParams['legend.scatterpoints'] = 1
    rcParams['legend.handlelength'] = 0.5
    rcParams['legend.handletextpad'] = 0.4

    # color cycle
    rcParams['axes.prop_cycle'] = cycler(color=vega_10)

    # axes
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.facecolor'] = 'white'

    # ticks (mpl default: k, k, medium, medium)
    rcParams['xtick.color'] = 'k'
    rcParams['ytick.color'] = 'k'
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize

    # axes grid (mpl default: False, #b0b0b0)
    rcParams['axes.grid'] = False
    rcParams['grid.color'] = '.8'

    # color map
    rcParams['image.cmap'] = 'RdBu_r' if color_map is None else color_map

    # frame (mpl default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon