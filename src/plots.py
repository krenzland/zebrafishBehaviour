# Set default settings for plotting.

import matplotlib.pyplot as plt
import seaborn as sns

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'text.usetex': True,
   'text.latex.preamble': [
        r'\usepackage{siunitx}',
        r'\sisetup{detect-all}',
        r'\usepackage[osf]{mathpazo}'
    ],
   'figure.figsize': [4.5, 4.5]
   }
plt.rcParams.update(params)

col_widths = {'paper': 312,
            'full': 468.3323,
            'margin': 144,
            'beamer': 307.28987}

# https://stackoverflow.com/a/31527287
def get_figsize(columnwidth=col_widths['paper'], wf=1, hf=(5.**0.5-1.0)/2.0, ):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]
