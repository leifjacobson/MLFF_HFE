


# MLFF_HFE

**MLFF_HFE** is a toolkit for computing hydration free energies (HFE) of organic molecules using machine-learned force fields (MLFF) with OpenMM, developed at [Schrodinger Inc](https://www.schrodinger.com/).


## Overview
This package provides scripts and utilities to run hydration free energy calculations using OpenMM and PyTorch-based force fields. The HFE computation follows the procedure described in:

> **Xie, X. et al.** "Accurate Hydration Free Energy Calculations for Diverse Organic Molecules With a Machine Learning Force Field."  
> [JCTC 2025](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02019)



## Installation

First, install [OpenMM](https://github.com/openmm/openmm), [openmmtorch](https://github.com/openmm/openmm-torch), and [openmmtools](https://github.com/choderalab/openmmtools) via conda:

```bash
conda config --add channels omnia --add channels conda-forge
conda install -c conda-forge openmm openmm-torch openmmtools 
```

Then install this package and its remaining dependencies:

```bash
pip install -e .
```

To install with test dependencies:

```bash
pip install .[test]
```


## Usage

### 1. Prepare a TorchScript Model

See [openmm-torch examples](https://github.com/openmm/openmm-torch) for how to export your model. Your `ForceModule`'s `forward` function should perform the linear mixing as defined in Equation (5) in the paper and have the following signature:

```python
class ForceModule(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor, cell: torch.Tensor, lambda_value_lj: torch.Tensor, lambda_value_nn: torch.Tensor, solute_temp: torch.Tensor) -> torch.Tensor:
        ... # returns energy
```

### 2. Run the Simulation

Example command:

```bash
python run_openmm_hfe.py \
    --pdb your_structure.pdb \
    --model model.pt \
    --lambda_schedule_lj 1.00,0.92,0.84,0.76,0.68,0.60,0.45,0.30,0.15,1.00,0.86,0.71,0.57,0.43,0.29,0.14,0.00 \
    --lambda_schedule_nn 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.14,0.29,0.43,0.57,0.71,0.86,1.00 \
    --lambda_schedule_solute_temp 298.15,346.84,403.48,469.38,546.03,635.20,738.94,859.62,1000.00,859.62,738.94,635.20,546.03,469.38,403.48,346.84,298.15
```



### 3. Output

The script produces a NetCDF file (default: `remd.nc`) containing the simulation results.


## License

Distributed under the terms of the [CC by 4.0](LICENSE).
