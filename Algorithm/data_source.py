"""Data source."""

import queue
import numpy as np
from typing import Optional

from qiskit.aqua.utils.validation import validate_min


class DataSource:
    """Data source."""

    def __init__(self, data: Optional[np.ndarray] = None, batch_size: int = 1):
        validate_min('batch_size', batch_size, 1)
        self.q = queue.Queue()
        self.batch_size = batch_size
        if data is not None:
            self.add_elements(data)

    def get_batch_size(self):
        return self.batch_size

    def is_empty(self):
        return self.q.empty()

    def serialized(self):
        return list(self.q.queue)

    def get_elements(self, size: int):
        assert(size > 0)
        batch = [self.q.get()]
        while len(batch) < size:
            batch.append(self.q.get())
        return batch

    def add_element(self, element):
        self.q.put(element)

    def add_elements(self, data: np.ndarray):
        for element in data:
            self.add_element(element)