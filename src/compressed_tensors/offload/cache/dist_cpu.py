# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import MutableMapping

import torch
import torch.distributed as dist
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.utils import send_tensors, to_empty

_BATCH_SIZE = 256


class DistributedCPUCache(CPUCache):
    """
    Handles offloading and onloading tensors from/to cpu memory shared across processes
    """

    def offload(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """
        Synchronously create shared cpu memory for offload

        :param tensor: tensor on any device
        :return: cpu tensor whose data is located in shared memory
        """
        if tensor is None:
            return None

        # slight runtime cost for views
        tensor = tensor.contiguous()

        if dist.get_rank() == 0:
            # create shared memory cpu tensor
            tensor = super().offload(tensor).share_memory_()
            (handle, filename, nbytes) = tensor.untyped_storage()._share_filename_cpu_()
            broadcast_obj = [handle, filename, nbytes]
        else:
            broadcast_obj = [None, None, None]

        # receive shared memory file handle
        dist.broadcast_object_list(broadcast_obj, src=0)

        if dist.get_rank() != 0:
            # materialize meta tensor only if necessary
            if tensor.device.type == "meta":
                tensor = to_empty(tensor, device=self.offload_device)
            else:
                tensor = send_tensors(tensor, device=self.offload_device)

            # reconstruct tensor from shared memory file handle
            with torch.no_grad():
                tensor.set_(
                    torch.UntypedStorage._new_shared_filename_cpu(*broadcast_obj),
                    storage_offset=tensor.storage_offset(),
                    size=tensor.size(),
                    stride=tensor.stride(),
                )

        # ensure that rank 0 does not garbage collect before other ranks reconstruct
        dist.barrier()

        return tensor

    @classmethod
    def from_mapping(
        cls,
        mapping: MutableMapping[str, torch.Tensor | None],
        onload_device: torch.device | str,
        **kwargs,
    ):
        # Distributed CPU offload is synchronization-bound, so batch handles by default.
        return cls.from_mapping_batched(mapping, onload_device, **kwargs)

    @classmethod
    def from_mapping_batched(
        cls,
        mapping,
        onload_device,
        batch_size=_BATCH_SIZE,
        **kwargs,
    ):
        """
        Batched version of from_mapping that groups broadcast + barrier
        operations to reduce NCCL overhead from O(num_tensors) to
        O(num_tensors / batch_size).

        Instead of broadcasting each tensor handle individually, we:
        1. Prepare all shared memory handles on rank 0
        2. Broadcast them in batches
        3. Reconstruct tensors on non-rank0 ranks
        4. Barrier once per batch instead of per tensor
        """
        if batch_size <= 0:
            raise ValueError(f"Expected positive batch_size, got {batch_size}")

        instance = cls(onload_device=onload_device, **kwargs)
        instance.offloaded_values = {}

        items = list(mapping.items())
        total = len(items)

        for batch_start in range(0, total, batch_size):
            batch_items = items[batch_start : batch_start + batch_size]
            tensors_prepared, received_handles = cls._prepare_offload_batch(
                instance, batch_items
            )
            cls._attach_shared_batch(instance, tensors_prepared, received_handles)

        return instance

    @staticmethod
    def _prepare_offload_batch(instance, batch_items):
        rank = dist.get_rank()
        handles_batch = []
        tensors_prepared = []

        for name, tensor in batch_items:
            if tensor is None:
                handles_batch.append(None)
                tensors_prepared.append((name, None))
                continue

            tensor = tensor.contiguous()

            if rank == 0:
                tensor = CPUCache.offload(instance, tensor).share_memory_()
                handle, filename, nbytes = tensor.untyped_storage()._share_filename_cpu_()
                handles_batch.append((handle, filename, nbytes))
            else:
                handles_batch.append(None)
                if tensor.device.type == "meta":
                    tensor = to_empty(tensor, device=instance.offload_device)
                else:
                    tensor = send_tensors(tensor, device=instance.offload_device)

            tensors_prepared.append((name, tensor))

        broadcast_obj = [handles_batch if rank == 0 else None]
        dist.broadcast_object_list(broadcast_obj, src=0)
        return tensors_prepared, broadcast_obj[0]

    @staticmethod
    def _attach_shared_batch(instance, tensors_prepared, received_handles):
        if dist.get_rank() != 0:
            with torch.no_grad():
                for idx, (_, tensor) in enumerate(tensors_prepared):
                    if tensor is None:
                        continue
                    handle = received_handles[idx]
                    tensor.set_(
                        torch.UntypedStorage._new_shared_filename_cpu(*handle),
                        storage_offset=tensor.storage_offset(),
                        size=tensor.size(),
                        stride=tensor.stride(),
                    )

        for name, tensor in tensors_prepared:
            instance.offloaded_values[name] = tensor

        # Keep rank 0 from releasing shared storage before peers attach to it.
        dist.barrier()
