# Copyright 2026 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict, deque
from math import ceil

import torch

from transformers.configuration_utils import PretrainedConfig


class CudaGraphBuffer:
    """A fixed-size dict for CUDA graphs with LRU eviction when full."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, but got {max_size}")
        self.max_size = max_size
        self._storage: OrderedDict[tuple[int, int], torch.cuda.CUDAGraph] = OrderedDict()

    def get_graph(self, q_len: int, kv_len: int) -> torch.cuda.CUDAGraph | None:
        graph = self._storage.get((q_len, kv_len))
        if graph is not None:
            self._storage.move_to_end((q_len, kv_len))
        return graph

    def set_graph(self, q_len: int, kv_len: int, graph: torch.cuda.CUDAGraph) -> None:
        if len(self._storage) >= self.max_size:
            _, graph = self._storage.popitem(last=False)
            graph.reset()
        self._storage[(q_len, kv_len)] = graph


class CpuGpuTimeTracker:

    def __init__(self, num_events: int = 8) -> None:
        self.cpu_queue = deque()
        self.gpu_queue = deque()
        self.gpu_compute_times = []
        self.cpu_prepare_times = []
        self.event_buffer = [torch.cuda.Event(enable_timing=True) for _ in range(num_events)]
        self.index = 0

    def get_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        start_event = self.event_buffer[self.index]
        self.index = (self.index + 1) % len(self.event_buffer)
        end_event = self.event_buffer[self.index]
        return start_event, end_event

    def start_cpu_span(self) -> None:
        self._start_span(for_cpu=True)

    def start_gpu_span(self) -> None:
        self._start_span(for_cpu=False)

    def _start_span(self, for_cpu: bool) -> None:
        start_event, end_event = self.get_event_pair()
        start_event.record()
        # Depending on the span type, select the correct variables
        queue_to_append = self.cpu_queue if for_cpu else self.gpu_queue
        queue_to_flush = self.gpu_queue if for_cpu else self.cpu_queue
        flush_destination = self.gpu_compute_times if for_cpu else self.cpu_prepare_times
        # Append the event pair to the correct queue
        queue_to_append.append((start_event, end_event))
        # Flush the other queue to get the elapsed time
        while queue_to_flush:
            event_pair = queue_to_flush.popleft()
            if event_pair[1].query():  # only flush if the end event is ready
                flush_destination.append(event_pair[1].elapsed_time(event_pair[0]))
            else:
                queue_to_flush.appendleft(event_pair)
                break

    def flush_timing_queues(self) -> None:
        for queue, dst in [(self.cpu_queue, self.cpu_prepare_times), (self.gpu_queue, self.gpu_compute_times)]:
            while queue:
                event_pair = queue.popleft()
                event_pair[1].wait()
                dst.append(event_pair[1].elapsed_time(event_pair[0]))

def attn_mask_is_needed(config: PretrainedConfig) -> bool:
    """Checks if attention mask is needed for the given (config)."""
    return config._attn_implementation in ["paged|eager", "paged|sdpa"]


def pad_to_interval(size: int, interval_size: int, max_value: int) -> int:
    """Return the smallest multiple of (interval_size) >= (size), capped at (max_value)."""
    if interval_size <= 0:
        return max_value
    padded = ceil(size / interval_size) * interval_size if size > 0 else interval_size
    return min(padded, max_value)


def aligned_divide(x: int, divide_by: int, align_to: int) -> int:
    x = int(ceil(x / divide_by))
    if x % align_to:
        x += align_to - (x % align_to)
    return x


def build_attention_mask(
    attention_mask: torch.Tensor,
    cumulative_seqlens_q: list[int],
    cumulative_seqlens_k: list[int],
    sliding_window: int = 1,
) -> None:
    """Builds an attention mask inplace using the cumulative seqlens of the query and key. If given a sliding window, it
    will also apply a sliding window mask on top. The attention mask is not boolean, it uses zeroes and -inf (or its
    equivalent) so it's more of an attention score bias tensor.
    The attention mask is a block-diagonal matrix, with each block an attention mask for a single query-key pair.
    Each of those block is built from a causal mask and, if there is a sliding window, a sliding window mask.

    An example is represented below, with seqlen_k = 8, seqlen_q = 4 and sliding_window = 6:

    CAUSAL MASK:

           █ █ █ █ █ ░ ░ ░
           █ █ █ █ █ █ ░ ░
           █ █ █ █ █ █ █ ░
           █ █ █ █ █ █ █ █

    SLIDING WINDOW MASK:
         ┌──────────────────────── seqlen_k - seqlen_q - sliding_window = 8 - 4 - 6 = -2 offset to the left
       <─┴─>
     ░ █ | █ █ █ █ █ █ █ █
     ░ ░ | █ █ █ █ █ █ █ █
     ░ ░ | ░ █ █ █ █ █ █ █
     ░ ░ | ░ ░ █ █ █ █ █ █

    ATTENTION MASK (sum of causal and sliding window masks):

           █ █ █ █ █ ░ ░ ░
           █ █ █ █ █ █ ░ ░
           ░ █ █ █ █ █ █ ░
           ░ ░ █ █ █ █ █ █

    Another example with seqlen_k = 5, seqlen_q = 3 and sliding_window = 2:

    CAUSAL MASK:

           █ █ █ ░ ░
           █ █ █ █ ░
           █ █ █ █ █

    SLIDING WINDOW MASK:
         ┌──────────────────────── seqlen_k - seqlen_q - sliding_window = 5 - 3 - 2 = 0 offset to the left
        <┴>
         | ░ █ █ █ █
         | ░ ░ █ █ █
         | ░ ░ ░ █ █

    ATTENTION MASK (sum of causal and sliding window masks):

           ░ █ █ ░ ░
           ░ ░ █ █ ░
           ░ ░ ░ █ █

    """
    min_value = torch.finfo(attention_mask.dtype).min
    for i in range(len(cumulative_seqlens_q) - 1):
        seqlen_q = cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i]
        seqlen_k = cumulative_seqlens_k[i + 1] - cumulative_seqlens_k[i]
        if seqlen_q < seqlen_k and seqlen_q >= 1:
            causal_diagonal = seqlen_k - seqlen_q + 1
        else:
            causal_diagonal = 1
        query_range = slice(cumulative_seqlens_q[i], cumulative_seqlens_q[i + 1])
        key_range = slice(cumulative_seqlens_k[i], cumulative_seqlens_k[i + 1])
        # Apply causal mask
        minus_inf = torch.full(
            attention_mask[..., query_range, key_range].shape,
            min_value,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        masked = torch.triu(minus_inf, diagonal=causal_diagonal)
        # Apply sliding window mask if needed
        if sliding_window > 1:
            sliding_diagonal = seqlen_k - seqlen_q - sliding_window
            masked += torch.tril(minus_inf, diagonal=sliding_diagonal)
        # Replace in attention mask
        attention_mask[..., query_range, key_range] = masked
