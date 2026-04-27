import argparse
import copy
import logging
import time

import numpy as np
import openmm
import torch
from ase.io import read
from openmm import MonteCarloBarostat
from openmm.app import PDBFile, Simulation, Topology
from openmm.unit import angstroms, atmosphere, femtosecond, kelvin
from openmmtools import mcmc, multistate
from openmmtools.multistate import ReplicaExchangeSampler
from openmmtools.states import (CompoundThermodynamicState,
                                GlobalParameterState, SamplerState,
                                ThermodynamicState)
from openmmtorch import TorchForce


def setup_logger() -> logging.Logger:
    """Configure and return a logger with DEBUG level output."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the HFE simulation.

    Returns:
        Parsed arguments including PDB file path, model path, lambda schedules,
        and simulation parameters (temperature, pressure, timestep, etc.).
    """
    parser = argparse.ArgumentParser(
        description="Run OpenMM HFE simulation with TorchForce.")
    parser.add_argument("--pdb",
                        type=str,
                        required=True,
                        help="Input PDB file")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Torch model file")
    parser.add_argument(
        "--lambda_schedule_lj",
        type=str,
        required=True,
        help="Comma-separated list for lambda_schedule_lj (e.g. 1,0.8,0.6)")
    parser.add_argument(
        "--lambda_schedule_nn",
        type=str,
        required=True,
        help="Comma-separated list for lambda_schedule_nn (e.g. 0,0.5,1)")
    parser.add_argument(
        "--lambda_schedule_solute_temp",
        type=str,
        required=True,
        help="Comma-separated list for lambda_schedule_solute_temp "
        "(e.g. 298.15,500,1000)")
    parser.add_argument("--output",
                        type=str,
                        default="remd.nc",
                        help="Output reporter file")
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Simulation temperature in Kelvin (default: 298.15)")
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Simulation pressure in atmospheres (default: 1.0)")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Integration timestep in femtoseconds (default: 0.5)")
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Langevin friction coefficient in 1/ps (default: 1.0)")
    parser.add_argument(
        "--pre_equilibration_steps",
        type=int,
        default=20000,
        help="Number of pre-equilibration steps per state (default: 20000)")
    parser.add_argument(
        "--remd_steps_per_iteration",
        type=int,
        default=1000,
        help="Number of MD steps per replica exchange attempt (default: 1000)")
    parser.add_argument(
        "--remd_num_iterations",
        type=int,
        default=500,
        help="Number of replica exchange attempts (default: 500)")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Checkpoint interval for REMD reporter (default: 1)")
    return parser.parse_args()


def setup_system(
        pdbfile_path: str, model_path: str
) -> tuple[openmm.System, TorchForce, PDBFile, Topology]:
    """Build an OpenMM System from a PDB file and a TorchForce model.

    Reads atomic structure via ASE, creates a bare System with particle masses
    and periodic box vectors, and initializes a TorchForce with placeholder
    global parameters. All pre-existing MM constraints and forces are removed
    so that only the machine-learned force field is used.

    Args:
        pdbfile_path: Path to the input PDB file.
        model_path: Path to the serialized TorchForce model file.

    Returns:
        A tuple of (system, force, pdbfile, topology).
    """
    atoms = read(pdbfile_path)
    force = TorchForce(model_path)
    force.addGlobalParameter('lambda_lj', 0.5)
    force.addGlobalParameter('lambda_nn', 0.3)
    force.addGlobalParameter('solute_temp', 400.0)
    force.setUsesPeriodicBoundaryConditions(True)

    pdbfile = PDBFile(pdbfile_path)
    topology = pdbfile.getTopology()

    system = openmm.System()
    for mass in atoms.get_masses():
        system.addParticle(mass)
    box_vectors = atoms.get_cell(complete=True) * angstroms
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # Remove MM constraints and forces
    while system.getNumConstraints() > 0:
        system.removeConstraint(0)
    while system.getNumForces() > 0:
        system.removeForce(0)

    if system.getNumConstraints() != 0:
        raise RuntimeError(
            f"Failed to remove all constraints from system. "
            f"Remaining constraints: {system.getNumConstraints()}")
    if system.getNumForces() != 0:
        raise RuntimeError(f"Failed to remove all forces from system. "
                           f"Remaining forces: {system.getNumForces()}")

    return system, force, pdbfile, topology


class LambdaState(GlobalParameterState):
    """Composable state managing alchemical lambda parameters.

    Tracks three global parameters for Hamiltonian replica exchange:
        lambda_lj: Lennard-Jones coupling parameter.
        lambda_nn: Neural network force field coupling parameter.
        solute_temp: Solute temperature for REST (Replica Exchange with
            Solute Tempering).
    """
    lambda_lj = GlobalParameterState.GlobalParameter('lambda_lj',
                                                     standard_value=1.0)
    lambda_nn = GlobalParameterState.GlobalParameter('lambda_nn',
                                                     standard_value=1.0)
    solute_temp = GlobalParameterState.GlobalParameter('solute_temp',
                                                       standard_value=500.0)

    def set_parameters(self, lambda_fep_lj: float, lambda_fep_nn: float,
                       solute_temp_fep: float) -> None:
        """Update all non-None alchemical parameters to the given values.

        Args:
            lambda_fep_lj: New value for the Lennard-Jones lambda parameter.
            lambda_fep_nn: New value for the neural network lambda parameter.
            solute_temp_fep: New value for the solute temperature parameter.
        """
        values = {
            'lambda_lj': lambda_fep_lj,
            'lambda_nn': lambda_fep_nn,
            'solute_temp': solute_temp_fep,
        }
        for name, value in values.items():
            if self._parameters.get(name) is not None:
                setattr(self, name, value)


def parse_lambda_schedule(schedule: str) -> np.ndarray:
    """Parse a comma-separated string into a NumPy array of floats.

    Args:
        schedule: Comma-separated string of numeric values

    Returns:
        Array of parsed float values.

    Raises:
        ValueError: If the schedule string is empty or contains invalid numbers.
    """
    values = [float(x) for x in schedule.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Lambda schedule is empty: '{schedule}'")
    return np.array(values)


def wrap_force_in_custom_cv(system: openmm.System, force: TorchForce) -> None:
    """Wrap a TorchForce in a CustomCVForce and add it to the system.

    This is a workaround required for creating a CompoundThermodynamicState
    with TorchForce. The TorchForce is wrapped in a CustomCVForce that
    exposes the same global parameters (lambda_lj, lambda_nn, solute_temp).

    See: https://github.com/openmm/openmm-torch/issues/147

    Args:
        system: The OpenMM System to add the wrapped force to.
        force: The TorchForce to wrap.
    """
    cv = openmm.CustomCVForce("")
    cv.addGlobalParameter("lambda_lj", 0.5)
    cv.addGlobalParameter("lambda_nn", 0.3)
    cv.addGlobalParameter("solute_temp", 400.0)
    temp_system = openmm.System()
    temp_system.addForce(force)
    var_names = []
    for idx, force_ in enumerate(temp_system.getForces()):
        name = f"allForce{idx+1}"
        cv.addCollectiveVariable(name, copy.deepcopy(force_))
        var_names.append(name)

    if len(var_names) == 0:
        raise RuntimeError(
            "No forces were added to the CustomCVForce. "
            "The TorchForce may not have been properly initialized.")

    cv.setEnergyFunction(f"({'+'.join(var_names)})")
    system.addForce(cv)


def run_pre_equilibration(
    compound_thermostate: CompoundThermodynamicState,
    topology: Topology,
    initial_positions: openmm.unit.Quantity,
    initial_box_vectors: openmm.unit.Quantity,
    lambda_schedule_lj: np.ndarray,
    lambda_schedule_nn: np.ndarray,
    lambda_schedule_solute_temp: np.ndarray,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[CompoundThermodynamicState], list[SamplerState]]:
    """Run short pre-equilibration MD for each alchemical lambda state.

    For each set of lambda values, creates a copy of the compound thermodynamic
    state, sets the alchemical parameters, and runs a short Langevin dynamics
    simulation to equilibrate the configuration.

    Args:
        compound_thermostate: Template thermodynamic state to copy for each replica.
        topology: System topology for creating Simulation objects.
        initial_positions: Starting particle positions.
        initial_box_vectors: Starting periodic box vectors.
        lambda_schedule_lj: Array of Lennard-Jones lambda values.
        lambda_schedule_nn: Array of neural network lambda values.
        lambda_schedule_solute_temp: Array of solute temperature values.
        args: Parsed command-line arguments containing temperature, friction,
            timestep, and pre_equilibration_steps.
        logger: Logger instance for status messages.

    Returns:
        A tuple of (thermostate_list, sampler_state_list) containing the
        equilibrated states for each replica.
    """
    sampler_state = SamplerState(initial_positions,
                                 box_vectors=initial_box_vectors)
    thermostate_list = []
    sampler_state_list = []
    logger.info("Starting pre-equilibration runs for all lambda states...")
    for i, (lj, nn, st) in enumerate(
            zip(lambda_schedule_lj, lambda_schedule_nn,
                lambda_schedule_solute_temp)):
        compound_thermostate_copy = copy.deepcopy(compound_thermostate)
        compound_thermostate_copy.set_parameters(lj, nn, st)
        logger.info(
            f'State {i}: lambda_lj={compound_thermostate_copy.lambda_lj}, '
            f'lambda_nn={compound_thermostate_copy.lambda_nn}, '
            f'solute_temp={compound_thermostate_copy.solute_temp}')

        eq_integrator = openmm.LangevinMiddleIntegrator(
            args.temperature, args.friction, args.timestep * femtosecond)
        simulation = Simulation(topology, compound_thermostate_copy.system,
                                eq_integrator)
        simulation.context.setParameter("lambda_lj",
                                        compound_thermostate_copy.lambda_lj)
        simulation.context.setParameter("lambda_nn",
                                        compound_thermostate_copy.lambda_nn)
        simulation.context.setParameter("solute_temp",
                                        compound_thermostate_copy.solute_temp)
        simulation.context.setPositions(initial_positions)
        simulation.context.setPeriodicBoxVectors(*initial_box_vectors)
        simulation.context.setVelocitiesToTemperature(args.temperature *
                                                      kelvin)
        sampler_state.apply_to_context(simulation.context)
        simulation.step(args.pre_equilibration_steps)
        sampler_state.update_from_context(simulation.context)
        thermostate_list.append(compound_thermostate_copy)
        sampler_state_list.append(copy.deepcopy(sampler_state))
        del eq_integrator
        del simulation
    return thermostate_list, sampler_state_list


def run_replica_exchange(
    thermostate_list: list[CompoundThermodynamicState],
    sampler_state_list: list[SamplerState],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Set up and run Hamiltonian Replica Exchange Molecular Dynamics.

    Args:
        thermostate_list: Pre-equilibrated thermodynamic states for each replica.
        sampler_state_list: Pre-equilibrated sampler states for each replica.
        args: Parsed command-line arguments containing timestep,
            remd_steps_per_iteration, remd_num_iterations, output path,
            and checkpoint_interval.
        logger: Logger instance for status messages.
    """
    move = mcmc.LangevinDynamicsMove(timestep=args.timestep * femtosecond,
                                     n_steps=args.remd_steps_per_iteration)
    sampler = ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=args.remd_num_iterations,
        replica_mixing_scheme='swap-all')
    reporter = multistate.MultiStateReporter(
        args.output, checkpoint_interval=args.checkpoint_interval)
    sampler.create(thermodynamic_states=thermostate_list,
                   sampler_states=sampler_state_list,
                   storage=reporter)
    logger.info("Starting Hamiltonian REMD run...")
    start_time = time.time()
    sampler.run()
    end_time = time.time()
    logger.info(f"Time to sample dynamics: {end_time - start_time:.2f} s")


def run_hfe_simulation(args: argparse.Namespace,
                       logger: logging.Logger) -> None:
    """Run a hydration free energy calculation using Hamiltonian REMD.

    Sets up the alchemical thermodynamic states from the provided lambda
    schedules, performs short pre-equilibration MD for each state, then
    runs Hamiltonian Replica Exchange Molecular Dynamics. Results are
    written to a NetCDF file via a MultiStateReporter.

    Args:
        args: Parsed command-line arguments containing file paths, lambda
            schedules, and simulation parameters.
        logger: Logger instance for status messages.
    """
    torch.set_printoptions(precision=10)
    logger.info("Setting up system...")
    system, force, pdbfile, topology = setup_system(args.pdb, args.model)

    wrap_force_in_custom_cv(system, force)

    barostat = MonteCarloBarostat(args.pressure * atmosphere,
                                  args.temperature * kelvin)
    system.addForce(barostat)

    lambda_state = LambdaState(lambda_lj=0.5, lambda_nn=0.3, solute_temp=500.0)
    thermostate = ThermodynamicState(system,
                                     temperature=args.temperature * kelvin)
    compound_thermostate = CompoundThermodynamicState(
        thermostate, composable_states=[lambda_state])

    lambda_schedule_lj = parse_lambda_schedule(args.lambda_schedule_lj)
    lambda_schedule_nn = parse_lambda_schedule(args.lambda_schedule_nn)
    lambda_schedule_solute_temp = parse_lambda_schedule(
        args.lambda_schedule_solute_temp)

    # Validate that all lambda schedules have the same length
    len_lj = len(lambda_schedule_lj)
    len_nn = len(lambda_schedule_nn)
    len_st = len(lambda_schedule_solute_temp)
    if not (len_lj == len_nn == len_st):
        raise ValueError("Lambda schedule length mismatch: "
                         f"lambda_schedule_lj (len={len_lj}), "
                         f"lambda_schedule_nn (len={len_nn}), "
                         f"lambda_schedule_solute_temp (len={len_st}) "
                         "must all have the same length.")
    logger.info(f'lambda_schedule_lj: {lambda_schedule_lj}')
    logger.info(f'lambda_schedule_nn: {lambda_schedule_nn}')
    logger.info(f'lambda_schedule_solute_temp: {lambda_schedule_solute_temp}')

    initial_box_vectors = system.getDefaultPeriodicBoxVectors()
    initial_positions = pdbfile.positions

    thermostate_list, sampler_state_list = run_pre_equilibration(
        compound_thermostate, topology, initial_positions, initial_box_vectors,
        lambda_schedule_lj, lambda_schedule_nn, lambda_schedule_solute_temp,
        args, logger)

    run_replica_exchange(thermostate_list, sampler_state_list, args, logger)


def main() -> None:
    """Entry point: parse arguments, set up logging, and run the simulation."""
    args = parse_args()
    logger = setup_logger()
    run_hfe_simulation(args, logger)


if __name__ == "__main__":
    main()
