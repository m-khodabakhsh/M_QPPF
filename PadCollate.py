from typing import Dict, Mapping, Sequence

import numpy as np
import torch


class PadCollate:
    def __init__(
        self,
        pad_token_id: int,
        pad_token_type_id: int,
    ):
        self._pad_token_id = pad_token_id
        self._pad_token_type_id = pad_token_type_id

    def get_pad_id(self, key: str) -> int:
        if key.endswith("input_ids"):
            return self._pad_token_id
        elif key.endswith("attention_mask"):
            return 0
        elif key.endswith("token_type_ids"):
            return self._pad_token_type_id
        else:
            assert False, f"Unknown key {key}"

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        from torch.utils.data.dataloader import default_collate  # type: ignore
        from irtools.pad import pad_to

        keys = {k for k, v in (batch[0])[0].items() if isinstance(v, torch.Tensor)}
        unalign = {k for k in keys if len(set((x[0])[k].size() for x in batch)) > 1}
        sizes = {k: np.max([(x[0])[k].size() for x in batch], axis=0) for k in unalign}
        data = [
            {
                k: pad_to(v, sizes[k], self.get_pad_id(k)) if k in unalign else v
                for k, v in x[0].items()
            }
            for x in batch
        ]
        queries =[
            x[1]
            for x in batch
        ]
        docSet = [
            x[2]
            for x in batch
        ]
        relevance_grades = [
            x[3]
            for x in batch
        ]
        MAPScore = [
            x[4]
            for x in batch
        ]
        collated: Dict[str, torch.Tensor] = default_collate(data)

        return collated, queries, docSet, relevance_grades, MAPScore
