class RateData:
    # It's like SpikeData but its underlying data is instaneous firing rates, not
    # sparse spike matrices.
    def __init__(
        self,
        inst_Frate_data,
        # subset_neurons = [],
        # subset_time_range = [],
        N=None,
        length=None,
    ):
        if inst_Frate_data.ndim != 2:
            raise ValueError(
                f"rates must be a 2D array, got shape {self.inst_Frate_data.shape}"
            )
        self.inst_Frate_data = inst_Frate_data

        self.N = inst_Frate_data.shape[0]
        self.length = inst_Frate_data.shape[1]
        self.subset_neurons = range(self.N)
        # Time is 0 indexed. So if someone says they want time 8, this refers to time 7-8.
        self.subset_time_range = (0, length - 1)

    def subset(self, units):
        """
        Extract a subset of neurons from the rate data.

        Parameters:
            units: List or array of neuron indices to extract

        Returns:
            RateData: New RateData object containing only the specified neurons
        """

        output = self.inst_Frate_data[units, :]
        # for neuron in units:
        #     neuron_firing_rate = self.inst_Frate_data[neuron,:]
        #     output.append(neuron_firing_rate)
        #     neuron_ids.append(neuron)
        return RateData(
            inst_Frate_data=output,
            subset_neurons=units,
            subset_time_range=self.subset_time_range,
        )

    def subtime(self, start, end):
        """
        Extract a subset of time points from the rate data.

        Parameters:
            start: Starting time index (inclusive)
            end: Ending time index (exclusive)

        Returns:
            RateData: New RateData object containing only the specified time range
        """
        output = self.inst_Frate_data[:, start:end]
        return RateData(
            inst_Frate_data=output,
            subset_neurons=self.subset_neurons,
            subset_time_range=(start, end),
        )
