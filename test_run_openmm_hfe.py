import logging
import sys

import numpy as np
import pytest

from run_openmm_hfe import (LambdaState, parse_args, parse_lambda_schedule,
                             setup_logger)


class TestParseLambdaSchedule:
    """Tests for parse_lambda_schedule."""

    def test_valid_input(self) -> None:
        result = parse_lambda_schedule("1.0,0.8,0.6,0.4")
        np.testing.assert_array_almost_equal(result, [1.0, 0.8, 0.6, 0.4])

    def test_single_value(self) -> None:
        result = parse_lambda_schedule("0.5")
        np.testing.assert_array_almost_equal(result, [0.5])

    def test_whitespace_handling(self) -> None:
        result = parse_lambda_schedule("1.0, 0.5, 0.0")
        np.testing.assert_array_almost_equal(result, [1.0, 0.5, 0.0])

    def test_trailing_comma(self) -> None:
        result = parse_lambda_schedule("1.0,0.5,")
        np.testing.assert_array_almost_equal(result, [1.0, 0.5])

    def test_returns_ndarray(self) -> None:
        result = parse_lambda_schedule("1,2,3")
        assert isinstance(result, np.ndarray)

    def test_integer_strings_become_floats(self) -> None:
        result = parse_lambda_schedule("1,2,3")
        assert result.dtype == np.float64


class TestParseArgs:
    """Tests for parse_args."""

    def test_required_args_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_openmm_hfe.py", "--pdb", "input.pdb", "--model", "model.pt",
            "--lambda_schedule_lj", "1,0.5",
            "--lambda_schedule_nn", "0,1",
            "--lambda_schedule_solute_temp", "298.15,500",
        ])
        args = parse_args()
        assert args.pdb == "input.pdb"
        assert args.model == "model.pt"
        assert args.output == "remd.nc"
        assert args.temperature == 298.15
        assert args.pressure == 1.0
        assert args.timestep == 0.5
        assert args.friction == 1.0
        assert args.pre_equilibration_steps == 20000
        assert args.remd_steps_per_iteration == 1000
        assert args.remd_num_iterations == 500
        assert args.checkpoint_interval == 1

    def test_custom_simulation_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_openmm_hfe.py", "--pdb", "input.pdb", "--model", "model.pt",
            "--lambda_schedule_lj", "1,0.5",
            "--lambda_schedule_nn", "0,1",
            "--lambda_schedule_solute_temp", "298.15,500",
            "--temperature", "310.0",
            "--pressure", "2.0",
            "--timestep", "1.0",
            "--friction", "0.5",
            "--pre_equilibration_steps", "10000",
            "--remd_steps_per_iteration", "500",
            "--remd_num_iterations", "100",
            "--checkpoint_interval", "5",
        ])
        args = parse_args()
        assert args.temperature == 310.0
        assert args.pressure == 2.0
        assert args.timestep == 1.0
        assert args.friction == 0.5
        assert args.pre_equilibration_steps == 10000
        assert args.remd_steps_per_iteration == 500
        assert args.remd_num_iterations == 100
        assert args.checkpoint_interval == 5

    def test_missing_required_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["run_openmm_hfe.py"])
        with pytest.raises(SystemExit):
            parse_args()


class TestSetupLogger:
    """Tests for setup_logger."""

    def test_returns_logger(self) -> None:
        logger = setup_logger()
        assert isinstance(logger, logging.Logger)

    def test_logger_level_is_debug(self) -> None:
        logger = setup_logger()
        assert logger.level == logging.DEBUG


class TestLambdaState:
    """Tests for LambdaState.set_parameters."""

    def test_set_parameters_updates_values(self) -> None:
        state = LambdaState(lambda_lj=1.0, lambda_nn=1.0, solute_temp=298.15)
        state.set_parameters(0.5, 0.3, 500.0)
        assert state.lambda_lj == 0.5
        assert state.lambda_nn == 0.3
        assert state.solute_temp == 500.0

    def test_set_parameters_different_values(self) -> None:
        state = LambdaState(lambda_lj=0.0, lambda_nn=0.0, solute_temp=298.15)
        state.set_parameters(1.0, 1.0, 1000.0)
        assert state.lambda_lj == 1.0
        assert state.lambda_nn == 1.0
        assert state.solute_temp == 1000.0
