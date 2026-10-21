"""The variant error the folder lookup still raises is a ValueError."""


def test_the_still_raised_variant_error_is_a_value_error():
    from multibench.engine.schema import AmbiguousVariantError
    assert issubclass(AmbiguousVariantError, ValueError)
