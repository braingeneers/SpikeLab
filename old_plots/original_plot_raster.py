
260115_plot_raster.py
Page
1
/
1
100%
def plot_raster_with_event(sd, time_range=None, fr_rates=None, event_times=None, event_boundaries=None,
                           sort_indices=None, font_size=14, save_path=None):
    """
    spk_mat and fr_rates both assumed to be Units x Time
    """

    # # # TO DO: MAKE THIS A SPIKEDATA METHOD (BUT SAVE SCRIPT IN SEPARATE plot_utils.py FILE)

    # # # TO DO: MAKE THE FONT SIZE CHANGES SPECIFIC TO FIGURE INSTEAD OF SETTING THEM GLOBALLY
    plt.rcParams.update({
        'font.size': font_size,  # Default for all text
        'axes.labelsize': font_size,  # x/y labels
        'xtick.labelsize': font_size,  # X tick labels
        'ytick.labelsize': font_size,  # Y tick labels
    })

    # obtain spike matrix from sd object
    spk_mat = sd.sparse_raster(bin_size=1).toarray()

    # if units should be reordered
    if sort_indices is not None:

        # reorder units in spike matrix
        spk_mat = spk_mat[sort_indices,:]

        # if firing rates are provided
        if fr_rates is not None: # # # TO DO: ALLOW USER TO HAVE fr_rates BE COMPUTED FROM SPIKEDATA INSTEAD OF PASSED

            # also reorder units in firing rate matrix
            fr_rates = fr_rates[sort_indices,:]

    # compute population rate
    smoothed_rate = sd.get_pop_rate(5, 5) # # # TO DO: 1. ENABLE USER TO PASS PRECOMPUTED POP_RATE 2. LET USER CHOOSE PARAMS FOR COMPUTING RATE

    # Filter spike times by optional time range
    if time_range is not None:

        # get start and end time
        start_time, end_time = time_range

        # cut spike matrix
        filtered_spk_mat = spk_mat[:, start_time:end_time]

        # cut smoothed_rate
        smoothed_rate = smoothed_rate[start_time:end_time]

        # filter event times for given time range
        if event_times is not None:
            event_mask = (event_times >= start_time) & (event_times <= end_time)
            event_times = event_times[event_mask] - start_time

        # cut firing rate matrix if provided
        if fr_rates is not None:
            fr_rates = fr_rates[:, start_time:end_time]

    else:
        start_time = 0
        end_time = spk_mat.shape[1]
        filtered_spk_mat = spk_mat

    # Prepare spikes for raster plot
    spike_times_list = []
    for i in range(filtered_spk_mat.shape[0]):
        spike_times_list.append(np.where(filtered_spk_mat[i, :] == 1)[0])

    # Initiate figure
    if fr_rates is None:

        plt.figure(figsize=(12, 6))
        # # # TO DO: 1. LET USER CHOOSE DIFFERENT FIGURE SIZE (ALSO FOR fr_rates IS NOT NONE) 2. LET USER CHOOSE
        # # # DIFFERENT SUBPLOT SIZE RATIOS

        # raster subplot
        ax1 = plt.subplot(2, 1, 1)

        # population rate subplot
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    else:

        plt.figure(figsize=(12, 8.5))

        # raster subplot
        ax1 = plt.subplot(3, 1, 1)

        # population rate subplot
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)

        # firing rate subplot
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

    # Plot rasterplot
    ax1.eventplot(spike_times_list, colors='black')
    ax1.set_ylim([0,spk_mat.shape[0]])
    ax1.set_ylabel('Unit')
    ax1.set_xlabel('Time (ms)')

    # Plot population firing rate
    ax2.plot(smoothed_rate, color='blue')
    ax2.set_ylabel('Population rate (Hz)')
    ax2.set_xlabel('Time (ms)')

    # if event times should be plotted
    if event_times is not None:

        # remove possible duplicate event times
        event_times = np.unique(event_times)

        # for each provided event time (if any)
        for e in event_times:

            # mark event
            ax2.scatter(e, smoothed_rate[e], c="k", zorder=9)

            # if event boundaries are included
            if event_boundaries is not None:

                # mark whole duration of event
                ax2.axvspan(e - event_boundaries[0], e + event_boundaries[1], color="b", alpha=0.2)
                # # # TO DO: SEPARATE PLOTTING EVENT PERIODS FROM EVENT TIMES. INSTEAD OF PLOTTING A PERIOD AROUND AN
                # # # EVENT TIME, LET THE USER PASS A LIST OF TUPLES WITH START AND END TIMES OF BOUNDARIES
                # # # IF WHOLE EVENT FALLS OUTSIDE OF time_range, IGNORE. IF START ONLY START OR END FALLS OUTSIDE OF
                # # # time_range, CLIP THE PERIOD TO THE END OF time_range. ADD THIS LOGIC TO FILTER SECTION AT START OF
                # # # FUNCTION


    # plot firing rates
    if fr_rates is not None:

        # # # TO DO: USE THE FUNCTION FROM 260115_plot_heatmap.py HERE
        
        # Plot heatmap of firing rates on ax3 with 'hot' colormap
        im = ax3.imshow(fr_rates, cmap='hot', aspect='auto', origin='lower', vmin=0, vmax=60)
        # # # TO DO: LET USER PASS vmin AND vmax VALUES (NONE BY DEFAULT). IF NO VALUES ARE PASSED, DO NOT SET THEM

        ax3.set_ylabel('Unit')
        ax3.set_xlabel('Time (ms)')

    # # # TO DO: SET X LIMITS FOR ALL AXES BASED ON time_range IF time_range IS NOT NONE

    # # # TO DO: LET THE USER PASS A BOOLEAN THAT INDICATES IF X-TICK LABELS SHOULD GO FROM time_range[0] TO
    # # # time_range[1] (DEFAULT BEHAVIOR) OR FROM 0 TO time_range[1] - time_range[0]

    plt.tight_layout()

    # TO DO: 1. RETURN FINAL FIGURE IN RETURN STATEMENT 2. ALLOW USER TO NOT PRINT OR SAVE FIGURE BUT ONLY HAVE IT RETURNED
    # Show or save figure
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
Displaying 260115_plot_raster.py.