import numpy as np


class RateData:
    # It's like SpikeData but its underlying data is instaneous firing rates, not
    # sparse spike matrices.

    # neuron_ids is a list of neuron_ids that each row index in inst_Frate_data represents
    # neuron_ids = [2,7,8] so inst_Frate_data has 3 rows, and row 0 represents neuron_id 2

    # times is a list of time values that each column index in inst_Frate_data represents
    # times = [5,10,15] so inst_Frate_data column 0 is 5 ms, column 1 is 10 ms, and column 2 is 15 ms
    def __init__(
        self,
        inst_Frate_data,
        neuron_ids,
        times,
        # subset_neurons = [],
        # subset_time_range = [],
        N=None,
        # length=None
    ):
        if inst_Frate_data.ndim != 2:
            raise ValueError(
                f"rates must be a 2D array, got shape {self.inst_Frate_data.shape}"
            )
        if len(neuron_ids) != inst_Frate_data.shape[0]:
            raise ValueError(
                "Number of rows in inst_Frate_data must be the same as length of neuron_ids"
            )
        if len(times) != inst_Frate_data.shape[1]:
            raise ValueError(
                "Number of columns in inst_Frate_data must be the same as length of times"
            )

        if any(x < 0 for x in times):
            raise ValueError("No negative values are allowed in times.")
        if not isinstance(times, np.ndarray):
            times = np.array(times)

        if not isinstance(neuron_ids, np.ndarray):
            neuron_ids = np.array(neuron_ids)

        self.inst_Frate_data = inst_Frate_data
        self.neuron_ids = neuron_ids
        self.times = times

        self.N = inst_Frate_data.shape[0]

        # self.length = times[-1] if len(times) > 0 else 0

        # self.subset_neurons = range(self.N)
        # # Time is 0 indexed. So if someone says they want time 8, this refers to time 7-8.
        # self.subset_time_range = (times[0], times[-1])

    def subset(self, units):
        """
        Extract a subset of neurons from the rate data.

        Parameters:
            units: List or array of neuron indices to extract (old)
            units: List or array of neuron_ids to extract

        Returns:
            RateData: New RateData object containing only the specified neurons
        """
        neuron_indices = []

        for i in range(len(units)):
            curr_neuron_id = units[i]
            for j in range(len(self.neuron_ids)):
                if self.neuron_ids[j] == curr_neuron_id:
                    neuron_indices.append(j)
        if len(neuron_indices) == 0:
            raise ValueError("Input Neuron_ids do not exist for this RateData Object")

        output = self.inst_Frate_data[neuron_indices, :]
        # for neuron in units:
        #     neuron_firing_rate = self.inst_Frate_data[neuron,:]
        #     output.append(neuron_firing_rate)
        #     neuron_ids.append(neuron)
        return RateData(inst_Frate_data=output, neuron_ids=units, times=self.times)

    def subtime(self, start, end):
        """
        Extract a subset of time points from the rate data.

        Parameters:
            start: Starting time index (inclusive)
            end: Ending time index (exclusive)

            start: Starting time  (inclusive)
            end: Ending time  (exclusive)

        Returns:
            RateData: New RateData object containing only the specified time range
        """

        # output = self.inst_Frate_data[:, start:end]
        # return RateData(
        #     inst_Frate_data=output,
        #     subset_neurons=self.subset_neurons,
        #     subset_time_range=(start, end),
        # )

        mask = (self.times >= start) & (self.times < end)

        # Check if start and end were in range
        if not np.any(mask):
            raise ValueError(
                f"No time points found in range [{start}, {end}). "
                f"The available range is [{self.times[0]}, {self.times[-1]}]"
            )

        output = self.inst_Frate_data[:, mask]
        new_times = self.times[mask]
        return RateData(
            inst_Frate_data=output,
            neuron_ids=self.neuron_ids,
            times=new_times,
        )
