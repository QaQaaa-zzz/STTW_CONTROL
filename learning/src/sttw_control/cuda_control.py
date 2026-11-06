"""Replay pure tensor kernels without changing their arithmetic."""
import torch


class CudaTensorFunction:
    def __init__(self, function, *example):
        self.inputs = [value.clone() for value in example]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):function(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = function(*self.inputs)

    def __call__(self, *args):
        for buffer, value in zip(self.inputs, args):buffer.copy_(value)
        self.graph.replay()
        if isinstance(self.output, tuple):return tuple(value.clone() for value in self.output)
        return self.output.clone()
