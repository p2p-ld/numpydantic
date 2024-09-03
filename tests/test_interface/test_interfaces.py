"""
Tests that should be applied to all interfaces
"""


def test_interface_revalidate(all_interfaces):
    """
    An interface should revalidate with the output of its initial validation

    See: https://github.com/p2p-ld/numpydantic/pull/14
    """
    _ = type(all_interfaces)(array=all_interfaces.array)
