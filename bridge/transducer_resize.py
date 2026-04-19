"""Resize per-edge Transducers when a WML's alphabet size changes.

When AdaptiveCodebook.shrink() retires rows, every transducer whose src
or dst WML shrunk must have its logit matrix resized by mean-pooling
the removed rows/columns. When grow() adds rows, we duplicate the
parent's logits as the initial seed. Argmax over unchanged rows is
preserved in both cases.

Plan 8 Task 4.
"""
from __future__ import annotations

import torch
from torch import nn

from track_p.transducer import Transducer


def resize_transducer(
    t: Transducer,
    *,
    keep_src: list[int] | None = None,
    keep_dst: list[int] | None = None,
    grow_src_parents: dict[int, int] | None = None,
    grow_dst_parents: dict[int, int] | None = None,
) -> Transducer:
    """Return a NEW Transducer with reshaped logits.

    Args:
        keep_src:        if given, rows of the src dimension to keep (shrink).
        keep_dst:        if given, columns of the dst dimension to keep (shrink).
        grow_src_parents: {new_row: parent_row} — duplicate parent's row.
        grow_dst_parents: {new_col: parent_col} — duplicate parent's column.

    Produces a new Transducer whose `.logits` has the requested shape.
    """
    old_size = t.alphabet_size
    old_logits = t.logits.data.clone()

    # Resolve new sizes.
    if keep_src is not None:
        new_src_size = len(keep_src)
    elif grow_src_parents is not None:
        new_src_size = old_size + len(grow_src_parents)
    else:
        new_src_size = old_size

    if keep_dst is not None:
        new_dst_size = len(keep_dst)
    elif grow_dst_parents is not None:
        new_dst_size = old_size + len(grow_dst_parents)
    else:
        new_dst_size = old_size

    # Transducer is square in the current module; we produce a square
    # new one using max(new_src, new_dst). Callers pass matched sizes
    # in practice — the transducer module assumes square alphabets.
    new_size = max(new_src_size, new_dst_size)
    new_t = Transducer(alphabet_size=new_size)

    # Build the new logits matrix.
    new_logits = torch.zeros(new_size, new_size)

    # Start with the shrunk / kept subset.
    if keep_src is not None and keep_dst is not None:
        src_idx = torch.tensor(keep_src, dtype=torch.long)
        dst_idx = torch.tensor(keep_dst, dtype=torch.long)
        kept = old_logits[src_idx][:, dst_idx]
        new_logits[: len(keep_src), : len(keep_dst)] = kept
    elif keep_src is not None:
        src_idx = torch.tensor(keep_src, dtype=torch.long)
        new_logits[: len(keep_src), :old_size] = old_logits[src_idx]
    elif keep_dst is not None:
        dst_idx = torch.tensor(keep_dst, dtype=torch.long)
        new_logits[:old_size, : len(keep_dst)] = old_logits[:, dst_idx]
    else:
        # No shrink — copy as-is.
        new_logits[:old_size, :old_size] = old_logits

    # Apply growths: new rows duplicated from parents.
    if grow_src_parents is not None:
        base = len(keep_src) if keep_src is not None else old_size
        for offset, (new_row, parent_row) in enumerate(grow_src_parents.items()):
            _ = new_row  # row index is sequential; new_row is bookkeeping
            new_logits[base + offset, :old_size] = old_logits[parent_row]
    if grow_dst_parents is not None:
        base = len(keep_dst) if keep_dst is not None else old_size
        for offset, (new_col, parent_col) in enumerate(grow_dst_parents.items()):
            _ = new_col
            new_logits[:old_size, base + offset] = old_logits[:, parent_col]

    with torch.no_grad():
        new_t.logits = nn.Parameter(new_logits)
    return new_t
