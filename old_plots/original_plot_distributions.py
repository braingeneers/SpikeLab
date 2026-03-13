
260115_plot_distributions.py
Page
1
/
1
100%
def plot_distributions(data_list, ax=None, x_tick_labs=None, font_size=14, save_path=None):

    # # # TO DO: LET USER CHOOSE BETWEEN PLOTTING DISTRIBUTIONS AS BOXPLOTS OR VIOLIN PLOTS

    # # # TO DO: LET USER CUSTOMIZE AXIS LABELS, FIG SIZE, VIOLIN/BOXPLOT COLORING, ETC.

    # # # TO DO: MAKE THE FONT SIZE CHANGES SPECIFIC TO FIGURE/AX INSTEAD OF SETTING THEM GLOBALLY
    plt.rcParams.update({
        'font.size': font_size,  # Default for all text
        'axes.labelsize': font_size,  # x/y labels
        'xtick.labelsize': font_size,  # X tick labels
        'ytick.labelsize': font_size,  # Y tick labels
    })
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        show_fig = True
    else:
        show_fig = False

    # get plot positions
    positions = range(1, len(data_list) + 1)

    # plot violins
    vplot = ax.violinplot(data_list, positions=positions, showmeans=False, showmedians=True)

    print("Total number of units:")
    for i, arr in enumerate(data_list):
        print("{}: {}".format(x_tick_labs[i], len(arr)))


    # # # BELOW IS LEFT IN FROM OLD FUNCTION AS STARTING POINT, MAKE MORE CUSTOMIZABLE
    # color first 2 violins one color, remaining another
    for i, body in enumerate(vplot['bodies']):
        if i < 2:
            body.set_facecolor('tab:blue')
        else:
            body.set_facecolor('tab:orange')
        body.set_edgecolor('black')
        body.set_alpha(0.8)
    # make all other lines (medians, bars, extrema) black
    for key in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
        if key in vplot and vplot[key] is not None:
            vplot[key].set_color('black')


    # Customize x-tick labels 
    if x_tick_labs is not None:
        ax.set_xticks(range(1, len(x_tick_labs) + 1), x_tick_labs)#, rotation=45, ha='right')

    # # # BELOW IS LEFT IN FROM OLD FUNCTION AS STARTING POINT, MAKE MORE CUSTOMIZABLE
    # Labels and styling
    ax.set_ylabel('Av. rate (Hz)')
    ax.set_yscale('log')
    ax.set_ylim(top=100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])

    # TO DO: 1. RETURN FINAL FIGURE OR AX IN RETURN STATEMENT 2. ALLOW USER TO NOT PRINT OR SAVE FIGURE BUT ONLY HAVE IT RETURNED
    if show_fig:
        if save_path is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
Displaying 260115_plot_distributions.py.