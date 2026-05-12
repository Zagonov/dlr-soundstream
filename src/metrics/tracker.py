class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._keys = list(keys)
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        self._data = {
            key: {
                "total": 0.0,
                "counts": 0.0,
                "average": 0.0,
            }
            for key in self._keys
        }

    def update(self, key, value, n=1):
        """
        Update metrics data with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        self._data[key]["total"] += float(value) * n
        self._data[key]["counts"] += n
        self._data[key]["average"] = (
            self._data[key]["total"] / self._data[key]["counts"]
        )

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data[key]["average"]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return {key: values["average"] for key, values in self._data.items()}

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (list[str]): all metric names.
        """
        return self._keys
