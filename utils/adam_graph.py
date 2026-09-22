"""Optional CUDA Graph for a repeatedly reused dense packet's Adam step.

Rendering/backward have dynamic shapes and are NOT captured. Native indexed
Adam is already a single kernel, so this graph is only a Torch fallback.
The captured code reads LR/bias corrections from fixed-address input tensors.
"""
import math
import torch


@torch.no_grad()
def tensor_adam_step(state, gradient, rates, corrections):
    d = state.shape[1] // 3
    p, m, v = state.split(d, dim=1)
    m.mul_(0.9).add_(gradient, alpha=0.1)
    v.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
    denominator = v.sqrt().div_(corrections[1]).add_(1e-8)
    p.sub_(m.div(denominator).mul_(rates / corrections[0]))


class PacketAdamGraph:
    def __init__(self, minimum_reuse=8):
        self.minimum_reuse = max(1, int(minimum_reuse))
        self.key = None
        self.reuse = 0
        self.graph = None
        self.inputs = None
        self.captures = self.replays = 0

    def reset(self):
        self.key = self.inputs = self.graph = None
        self.reuse = 0

    @torch.no_grad()
    def step(self, packet, grad, rates, iteration, frozen_prefix=0):
        from utils.resident_pool_v2 import IndexedPacket
        if isinstance(packet, IndexedPacket) or not packet.state.is_cuda:
            return False
        key = (packet.state.data_ptr(), tuple(packet.state.shape), frozen_prefix)
        if key != self.key:
            self.reset()
            self.key = key
        self.reuse += 1
        if self.reuse < self.minimum_reuse:
            return False
        if self.graph is None:
            # Fixed buffers, no replay-time Python scalars and no shape-changing
            # ops. Capture itself does not apply an optimizer step.
            gradient = torch.empty_like(grad)
            rate_input = torch.empty_like(rates)
            corrections = torch.empty(2, device=grad.device)
            frozen = torch.as_tensor(packet.ids < frozen_prefix, device=grad.device)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=grad.device)
            stream.wait_stream(torch.cuda.current_stream(grad.device))
            with torch.cuda.graph(graph, stream=stream):
                masked = gradient.masked_fill(frozen[:, None], 0)
                tensor_adam_step(packet.state, masked, rate_input, corrections)
            torch.cuda.current_stream(grad.device).wait_stream(stream)
            self.graph = graph
            # Hold the owning state tensor; a stale allocator address is not a
            # valid cache key once the previous packet has been destroyed.
            self.inputs = (gradient, rate_input, corrections, packet.state, frozen)
            # The captured frozen mask is an external tensor; retain it across replays.
            self.captures += 1
        gradient, rate_input, corrections, _, _frozen = self.inputs
        gradient.copy_(grad)
        rate_input.copy_(rates)
        corrections.copy_(torch.tensor([1 - 0.9**(iteration + 1), math.sqrt(1 - 0.999**(iteration + 1))],
                                        dtype=torch.float32, device=grad.device))
        self.graph.replay()
        packet.dirty = True
        self.replays += 1
        return True
