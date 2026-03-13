
260115_plot_heatmap.py
Page
1
/
1
100%
def plot_av_rate(data_mat, ax=None, norm=False, colorbar_label="Av. Rate (Hz)", temporal_offset=0, vmax=40,
                 font_size=14, save_path=None):

    # # # TO DO: MAKE THIS FUNCTION MORE GENERAL SO THAT IT CAN BE USED FOR PLOTTING ANY TYPE OF HEATMAP

    # # # TO DO: MAKE THIS A METHOD OF RateData (BUT SAVE SCRIPT IN SEPARATE plot_utils.py FILE)

    # # # TO DO: MAKE THE FONT SIZE CHANGES SPECIFIC TO FIGURE/AX INSTEAD OF SETTING THEM GLOBALLY
    plt.rcParams.update({
        'font.size': font_size,  # Default for all text
        'axes.labelsize': font_size,  # x/y labels
        'xtick.labelsize': font_size,  # X tick labels
        'ytick.labelsize': font_size,  # Y tick labels
    })
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fig = True
    else:
        plot_fig = False

    # # # TO DO: LET USER CHOOSE BETWEEN NORMALIZING PER ROW (AS BELOW) PER COLUMN (TO BE ADDED) OR NOT AT ALL (DEFAULT)
    if norm:
        normalized_data = np.zeros_like(data_mat)
        for i in range(data_mat.shape[0]):
            row_min = data_mat[i].min()
            row_max = data_mat[i].max()
            if row_max > row_min:
                normalized_data[i] = (data_mat[i] - row_min) / (row_max - row_min)
            else:
                normalized_data[i] = data_mat[i]  # Handle constant rows
        data_mat = normalized_data
        colorbar_label = "Norm. " + colorbar_label
        vmax=1
        
    # # # TO DO: LET USER PASS vmin AND vmax VALUES (NONE BY DEFAULT). IF NO VALUES ARE PASSED, DO NOT SET THEM
    im = ax.imshow(data_mat, cmap='hot', vmin=0, vmax=vmax, aspect='auto')

    # # # TO DO: LET USER CHOOSE AXIS LABELS
    ax.set_xlabel('Relative time (ms)')
    ax.set_ylabel('Unit')

    # # # TO DO: LET USER PASS X AND Y TICK LOCATIONS AND LABELS OPTIONALLY
    ax.set_yticks([1, data_mat.shape[0]])

    # Subtract temporal_offset from each x-tick label
    xtick_labels = [str(int(tick) - temporal_offset) for tick in ax.get_xticks()]
    ax.set_xticklabels(xtick_labels)

    # # # TO DO: ALLOW USER TO OPTIONALLY PASS ONE OR MULTIPLE X AND/OR Y LINES INSTEAD OF ONLY THIS SINGLE X LINE
    # # # ALSO GIVE MORE CONTROL OVER COLOR, STYLE AND WIDTH
    # Add vertical dotted green line at temporal_offset
    ax.axvline(x=temporal_offset, color='green', linestyle='dotted', linewidth=2)

    # ALLOW USER TO CHOOSE NOT TO PLOT COLORBAR
    plt.colorbar(im, ax=ax, label=colorbar_label)

    # TO DO: 1. RETURN FINAL FIGURE OR AX IN RETURN STATEMENT 2. ALLOW USER TO NOT PRINT OR SAVE FIGURE BUT ONLY HAVE IT RETURNED
    if plot_fig:
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()
Displaying 260115_plot_heatmap.py.