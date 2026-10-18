"""Your rows of a package method carry its language chip, '?' marks a name the package
does not know, and the grey-column warning says what to do (core040 F-3, F-4)."""
from multibench.plot.bubble import _method_language, _no_comparison_message


def test_this_run_rows_have_the_methods_language():
    assert _method_language("Matilda (this run)") == _method_language("Matilda") == "python"
    assert _method_language("My PCA") is None


def test_the_grey_column_warning_says_how_to_leave_it_out():
    msg = _no_comparison_message(["A", "B"], {"cLISI": 1.0})
    assert msg.endswith("That column is grey. Leave it out with metrics=.")
