"""
Tests for dunder methods on all interfaces
"""


def test_dunder_len(all_interfaces):
    """
    Each interface or proxy type should support __len__
    """
    assert len(all_interfaces.array) == all_interfaces.array.shape[0]
