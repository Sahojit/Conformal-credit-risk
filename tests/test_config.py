import pytest

from conformal_credit_risk.config import ConformalConfig, DataConfig, ExperimentConfig


def test_default_config_loads():
    config = ExperimentConfig()
    assert config.data.train_fraction + config.data.calibration_fraction + config.data.test_fraction == pytest.approx(1.0)


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataConfig(train_fraction=0.5, calibration_fraction=0.3, test_fraction=0.3)


def test_coverage_levels_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConformalConfig(coverage_levels=[0.9, 1.5])


def test_default_coverage_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ConformalConfig(default_coverage=0.0)
