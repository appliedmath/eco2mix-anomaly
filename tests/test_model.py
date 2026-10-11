import pandas as pd

from src.models.model import (
    fit_bias,
    fit_reference,
    flag_with_reference,
    load_model,
    save_model,
)


def test_fit_bias_is_median_of_errors():
    errors = pd.Series([-10.0, -20.0, -30.0, 100.0, -40.0])
    # sorted: -40, -30, -20, -10, 100 -> the middle one is -20
    assert fit_bias(errors) == -20.0


def test_fit_bias_ignores_missing_values():
    errors = pd.Series([1.0, None, 3.0])
    assert fit_bias(errors) == 2.0


def test_fit_reference_median_and_mad():
    ref = fit_reference(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    # median = 3; distances to it: 2, 1, 0, 1, 2 -> their median = 1
    assert ref == {"median": 3.0, "mad": 1.0}


def test_flag_with_reference_flags_only_far_gaps():
    ref = {"median": 0.0, "mad": 1.0}
    actual = pd.Series([0.0, 1.0, 100.0])
    expected = pd.Series([0.0, 0.0, 0.0])
    flags = flag_with_reference(actual, expected, ref, k=4.0)
    assert flags.tolist() == [False, False, True]


def test_flag_with_reference_uses_learned_median():
    # A gap of -100 is "normal" when the learned median gap is -100
    ref = {"median": -100.0, "mad": 10.0}
    actual = pd.Series([-100.0, 0.0])
    expected = pd.Series([0.0, 0.0])
    flags = flag_with_reference(actual, expected, ref, k=4.0)
    assert flags.tolist() == [False, True]


def test_save_and_load_roundtrip(tmp_path):
    model = {"type": "rte_plus_bias", "bias": -102.5}
    meta = {"k": 4.0, "mad": 449.0}
    save_model(model, meta, folder=str(tmp_path))
    loaded_model, loaded_meta = load_model(folder=str(tmp_path))
    assert loaded_model == model
    assert loaded_meta == meta

