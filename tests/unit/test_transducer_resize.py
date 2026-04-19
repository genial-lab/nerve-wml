"""Tests for bridge.transducer_resize."""
import torch

from bridge.transducer_resize import resize_transducer
from track_p.transducer import Transducer


def test_shrink_produces_smaller_transducer():
    t = Transducer(alphabet_size=16)
    keep = list(range(8))
    new_t = resize_transducer(t, keep_src=keep, keep_dst=keep)
    # The new transducer is a square max-of-sizes matrix; kept subset fills top-left.
    assert new_t.logits.shape == (8, 8)


def test_shrink_preserves_argmax_on_kept_rows():
    """If we keep rows [0, 2, 4] and columns [0, 2, 4], the argmax of row 0
    in the new transducer must equal the argmax of row 0 restricted to those
    columns in the old transducer."""
    t = Transducer(alphabet_size=16)
    t.logits.data = torch.randn(16, 16)

    keep = [0, 2, 4, 6, 8, 10, 12, 14]
    new_t = resize_transducer(t, keep_src=keep, keep_dst=keep)
    # Row 0 of new_t argmax, compared to row 0 of old_t restricted to keep cols.
    old_row0_restricted = t.logits.data[0, keep]
    assert new_t.logits.data[0].argmax().item() == old_row0_restricted.argmax().item()


def test_grow_produces_larger_transducer():
    t = Transducer(alphabet_size=8)
    grow = {8: 0, 9: 2}  # new rows 8, 9 inherit from parents 0, 2
    new_t = resize_transducer(t, grow_src_parents=grow, grow_dst_parents=grow)
    assert new_t.logits.shape == (10, 10)


def test_grow_copies_parent_row_into_child():
    t = Transducer(alphabet_size=8)
    t.logits.data = torch.randn(8, 8)

    grow = {8: 0}
    new_t = resize_transducer(t, grow_src_parents=grow)
    # New size is max(9, 8) = 9. Row 8 should equal old row 0 on the first 8 cols.
    assert new_t.logits.shape == (9, 9)
    assert torch.allclose(new_t.logits.data[8, :8], t.logits.data[0])


def test_no_resize_copies_logits_as_is():
    """Neither keep nor grow → identity copy."""
    t = Transducer(alphabet_size=8)
    t.logits.data = torch.randn(8, 8)
    new_t = resize_transducer(t)
    assert new_t.logits.shape == t.logits.shape
    assert torch.allclose(new_t.logits.data, t.logits.data)
