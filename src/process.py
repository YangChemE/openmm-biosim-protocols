import sys 
import os
import argparse
import io

import copy
from pathlib import Path
from Bio import PDB
import requests
import logging
#from IPython.display import display
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import mdtraj as md
import MDAnalysis as mda 
from MDAnalysis import transformations
from MDAnalysis.analysis import rms, contacts, align
from MDAnalysis.analysis.align import rotation_matrix
import MDAnalysis.analysis.encore as encore
import prolif as plf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import pdbfixer
import openmm as mm
import openmm.app as app
from openmm import *
from openmm.app import *
from openmm import unit
from openmm.app.metadynamics import *
from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import GAFFTemplateGenerator


def remove_protein_hydrogens(input_pdb, output_pdb):

    # Load the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', input_pdb)

    # Select atoms to keep (non-hydrogen atoms of protein and all atoms of ligand)
    io = PDB.PDBIO()
    class NonHetSelect(PDB.Select):
        def accept_atom(self, atom):
            if atom.element == 'H' and atom.get_parent().get_resname() != 'MOL':
                return False
            return True

    # Save the modified structure to a new PDB file
    io.set_structure(structure)
    io.save(output_pdb, NonHetSelect())
    
def get_pdb(pdbid, save_addr):
    pdb_url = f"https://files.rcsb.org/download/{pdbid}.pdb"
    r = requests.get(pdb_url)
    r.raise_for_status()
    with open(save_addr, 'w') as f:
        f.write(r.text)
        
def prepare_protein(pdb_file, ignore_missing_residues = True, ignore_terminal_missing_residues = True, ph = 7.0):
    """
    Use pdbfixer to prepare the protein from a PDB file. Hetero atoms such as ligands are
    removed and non-standard residues replaced. Missing atoms to existing residues are added.
    Missing residues are ignored by default, but can be included.

    Parameters
    ----------
    pdb_file: pathlib.Path or str
        PDB file containing the system to simulate.
    ignore_missing_residues: bool, optional
        If missing residues should be ignored or built.
    ignore_terminal_missing_residues: bool, optional
        If missing residues at the beginning and the end of a chain should be ignored or built.
    ph: float, optional
        pH value used to determine protonation state of residues

    Returns
    -------
    fixer: pdbfixer.pdbfixer.PDBFixer
        Prepared protein system.
    """
    
    fixer = pdbfixer.PDBFixer(str(pdb_file))
    fixer.removeHeterogens()  # co-crystallized ligands are unknown to PDBFixer
    fixer.findMissingResidues()  # identify missing residues, needed for identification of missing atoms

    # if missing terminal residues shall be ignored, remove them from the dictionary
    if ignore_terminal_missing_residues:
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    # if all missing residues shall be ignored ignored, clear the dictionary
    if ignore_missing_residues:
        fixer.missingResidues = {}
    
    fixer.findNonstandardResidues()  # find non-standard residue
    fixer.replaceNonstandardResidues()  # replace non-standard residues with standard one
    fixer.findMissingAtoms()  # find missing heavy atoms
    fixer.addMissingAtoms()  # add missing atoms and residues
    fixer.addMissingHydrogens(ph)  # add missing hydrogens
    return fixer


def prepare_ligand(pdb_file, resname, sdf, depict = False):
    """
    Prepare a ligand from a PDB file via adding hydrogens and assigning bond orders. A depiction
    of the ligand before and after preparation is rendered in 2D to allow an inspection of the
    results. Huge thanks to @j-wags for the suggestion.

    Parameters
    ----------
    pdb_file: pathlib.PosixPath
       PDB file containing the ligand of interest.
    resname: str
        Three character residue name of the ligand.
    sdf : str
        The generated sdf file for assigned bond order
        SMILES string of the ligand informing about correct protonation and bond orders.
    depict: bool, optional
        show a 2D representation of the ligand

    Returns
    -------
    prepared_ligand: rdkit.Chem.rdchem.Mol
        Prepared ligand.
    """
    # split molecule
    rdkit_mol = Chem.MolFromPDBFile(str(pdb_file))
    rdkit_mol_split = Chem.rdmolops.SplitMolByPDBResidues(rdkit_mol)

    # extract the ligand and remove any already present hydrogens
    ligand = rdkit_mol_split[resname]
    ligand = Chem.RemoveHs(ligand)

    # assign bond orders from template
    reference_mol = Chem.SDMolSupplier(sdf)[0]
    prepared_ligand = AllChem.AssignBondOrdersFromTemplate(reference_mol, ligand)
    prepared_ligand.AddConformer(ligand.GetConformer(0))

    # protonate ligand
    prepared_ligand = Chem.rdmolops.AddHs(prepared_ligand, addCoords=True)
    prepared_ligand = Chem.MolFromMolBlock(Chem.MolToMolBlock(prepared_ligand))


    # 2D depiction
    if depict:
        ligand_2d = copy.deepcopy(ligand)
        prepared_ligand_2d = copy.deepcopy(prepared_ligand)
        AllChem.Compute2DCoords(ligand_2d)
        AllChem.Compute2DCoords(prepared_ligand_2d)
        display(
            Draw.MolsToGridImage(
                [ligand_2d, prepared_ligand_2d], molsPerRow=2, legends=["original", "prepared"]
            )
        )
    
    return prepared_ligand

def rdkit_to_openmm(rdkit_mol, name = 'LIG'):
    """
    Convert an RDKit molecule to an OpenMM molecule.
    Inspired by @hannahbrucemcdonald and @glass-w.

    Parameters
    ----------
    rdkit_mol: rdkit.Chem.rdchem.Mol
        RDKit molecule to convert.
    name: str
        Molecule name.

    Returns
    -------
    omm_molecule: openmm.app.Modeller
        OpenMM modeller object holding the molecule of interest.
    """
    
    # convert RDKit to OpenFF
    off_mol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo = True)
    
    # add name for molecule 
    off_mol.name = name
    
    # add names for atoms
    element_counter_dict = {}
    for off_atom, rdkit_atom in zip(off_mol.atoms, rdkit_mol.GetAtoms()):
        element = rdkit_atom.GetSymbol()
        if element in element_counter_dict.keys():
            element_counter_dict[element] += 1
        else:
            element_counter_dict[element] = 1
        off_atom.name = element + str(element_counter_dict[element])

    # convert from OpenFF to OpenMM
    off_mol_topology = off_mol.to_topology()
    mol_topology = off_mol_topology.to_openmm()
    mol_positions = off_mol.conformers[0]
    
    # convert units from Ångström to nanometers
    # since OpenMM works in nm
    mol_positions = mol_positions.to("nanometers")
    
    # combine topology and positions in modeller object 
    omm_mol = app.Modeller(mol_topology, mol_positions)
    
    return omm_mol


def merge_protein_and_ligand(protein, ligand):
    """
    Merge two OpenMM objects.

    Parameters
    ----------
    protein: pdbfixer.pdbfixer.PDBFixer
        Protein to merge.
    ligand: openmm.app.Modeller
        Ligand to merge.

    Returns
    -------
    complex_topology: openmm.app.topology.Topology
        The merged topology.
    complex_positions: openmm.unit.quantity.Quantity
        The merged positions.
    """
    
    # combine topologies
    md_protein_topology = md.Topology.from_openmm(protein.topology)  # using mdtraj for protein top
    md_ligand_topology = md.Topology.from_openmm(ligand.topology)  # using mdtraj for ligand top
    md_complex_topology = md_protein_topology.join(md_ligand_topology)  # add them together
    complex_topology = md_complex_topology.to_openmm()
    
    
    # combine positions 
    total_atoms = len(protein.positions) + len(ligand.positions)
    
    # create an array for storing all atom positions as tupels containing a value and a unit
    # called OpenMM Quantities
    complex_positions = unit.Quantity(np.zeros([total_atoms, 3]), unit=unit.nanometers)
    complex_positions[: len(protein.positions)] = protein.positions  # add protein positions
    complex_positions[len(protein.positions) :] = ligand.positions  # add ligand positions
    
    # Set the chain id in the md_complex_topology based on the protein.topology
    protein_chain_count = len(list(protein.topology.chains()))
    for chn_new, chn_ori in zip(list(complex_topology.chains())[:protein_chain_count], protein.topology.chains()):
        chn_new.id = chn_ori.id
    
    for chn in list(complex_topology.chains())[protein_chain_count:]:
        chn.id = 'X'
    
    return complex_topology, complex_positions, md_complex_topology
 
def generate_forcefield(rdkit_mol = None, protein_ff = "amber14-all.xml", solvent_ff = "amber14/tip3pfb.xml"):
    
    """
    Generate an OpenMM Forcefield object and register a small molecule.

    Parameters
    ----------
    rdkit_mol: rdkit.Chem.rdchem.Mol
        Small molecule to register in the force field.
    protein_ff: string
        Name of the force field.
    solvent_ff: string
        Name of the solvent force field.

    Returns
    -------
    forcefield: openmm.app.Forcefield
        Forcefield with registered small molecule.
    """
    
    forcefield = app.ForceField(protein_ff, solvent_ff)
    forcefield_vac = app.ForceField(protein_ff)
    
    if rdkit_mol is not None:
        off_mol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo = True)
        try:
            off_mol.assign_partial_charges(partial_charge_method="am1bcc")
        except:
            print(f"sqm can't be run properly, using mmff94 charge instead off am1bcc.\n")
            off_mol.assign_partial_charges(partial_charge_method="mmff94")
            
        gaff = GAFFTemplateGenerator(molecules = off_mol)
        forcefield.registerTemplateGenerator(gaff.generator)
        forcefield_vac.registerTemplateGenerator(gaff.generator)
        
    return forcefield, forcefield_vac
    

def get_anchor_ca(min_pdb, ligand_name = 'UNK'):
    """
    Get the coordinates of the C-alpha atom of the anchor residue.

    Parameters
    ----------
    min_pdb: str
        Path to the minimized PDB file.
    
    ligand_name: str
        Name of the ligand in the PDB file.

    Returns
    -------
    anchor_ca: np.array
        Coordinates of the C-alpha atom of the anchor residue.
    """
    
    # load the minimized structure
    u = mda.Universe(min_pdb, format='XPDB', in_memory=True)
    mol = u.select_atoms(f"resname {ligand_name}")
    pkt = u.select_atoms(f"protein and around 6 resname {ligand_name}")
    
    mol_COM = mol.center_of_geometry()
    pkt_COM = pkt.center_of_geometry()
    
    outward_v = mol_COM - pkt_COM
    pkt_ca = u.select_atoms(f"name CA and around 5 resname {ligand_name}")
    
    best_ca = None 
    max_projection = -np.inf

    for ca in pkt_ca:
        ca_v = mol_COM - ca.position
        projection = np.dot(ca_v, outward_v) 
        if projection > max_projection:
            best_ca = ca
    
    anchor_index = best_ca.id - 1
    
    return anchor_index

    
def get_pose_rmsd(ref_pdb, md_trj, anchor_index, start = 0, end = -1, ligand_name = 'UNK',):
    """
    Calculate the RMSD of a ligand in a trajectory to a reference structure.

    Parameters
    ----------
    ref_pdb: str
        Path to the reference PDB file.
    md_trj: str
        Path to the trajectory file.
    anchor_index: int 
        Index of the anchor ca atom
    start: int, optional
        Start frame of the trajectory, default is 0.
    end: int, optional
        End frame of the trajectory, default is -1.
    ligand_name: str
        Name of the ligand in the PDB file.
    Returns
    -------
    rmsd: float
        RMSD of the ligand in the trajectory.
    """
    
    # load reference structure
    ref = mda.Universe(ref_pdb)

    # load trajectory
    trj = mda.Universe(ref_pdb, md_trj)
    sel = f"resname {ligand_name} and not name H* or index {anchor_index}"
    
    
    # calculate RMSD
    r = rms.RMSD(trj, ref, select = sel, groupselections = [sel])
    r.run(start = start, stop = end)
    rmsd = r.rmsd[:,2] 
    return rmsd


def minimization(ff, top, pos, out_dir, min_pdb_name, use_gpu = True):
    """
    Minimize the energy of a system.
    
    Parameters
    ----------
    ff: openmm.app.ForceField
        Forcefield object.
    top: openmm.app.topology.Topology
        Topology object.
    pos: list
        List of positions.
    out_dir: str
        Output directory for the minimized structure.
    min_pdb_name: str
        Name of the minimized PDB file.
    use_gpu: bool, optional
        If or not use GPU for the minimization, default is True.
    Returns
    -------
    None
    """
    
    ## create system
    system = ff.createSystem(
        topology = top,
        nonbondedMethod=app.PME,
        constraints=HBonds,
        hydrogenMass=4*unit.amu,
        rigidWater = True
    )
    
    
    # Define platform properties
    if use_gpu:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
    else:
        platform = Platform.getPlatformByName('CPU')
        properties = {}
    
    ## define integrator
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    ## create simulation
    simulation = app.Simulation(top, system, integrator, platform, properties)
    simulation.context.setPositions(pos)
    
    ## run minimization
    simulation.minimizeEnergy()
    
    ## save minimized structure
    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox = True).getPositions()
    out_file = os.path.join(out_dir,min_pdb_name)
    PDBFile.writeFile(simulation.topology, positions,
                      open(out_file, 'w'),keepIds=True)
    
    
    return None

def nvt_equilibration(min_pdb, ff, top, out_dir, eq_pdb_name):
    """
    Equilibrate the system.
    
    Parameters
    ----------
    min_pdb: str
        Path to the minimized PDB file.
    ff: openmm.app.ForceField
        Forcefield object.
    top: openmm.app.topology.Topology
        Topology object.
    out_dir: str
        Output directory for the equilibrated structure.
    eq_pdb_name: str
        Name of the equilibrated PDB file.
        
    Returns
    -------
    None
    """
    
    # Get the solute heavy atom indices to use
    # for defining position restraints during equilibration
    universe = mda.Universe(min_pdb, format='XPDB', in_memory=True)
    solute_heavy_atom_idx = universe.select_atoms('not resname WAT and\
                                                   not resname SOL and\
                                                   not resname HOH and\
                                                   not resname CL and \
                                                   not resname NA and \
                                                   not name H*').indices
    
    # Necessary conversion to int from numpy.int64,
    # b/c it breaks OpenMM C++ function
    solute_heavy_atom_idx = [int(idx) for idx in solute_heavy_atom_idx]
    
    
    ## create system
    system = ff.createSystem(
        topology = top,
        nonbondedMethod=app.PME,
        constraints=HBonds,
        hydrogenMass=4*unit.amu,
        rigidWater = True
    )
    
    # Add the restraints.
    # We add a dummy atoms with no mass, which are therefore unaffected by
    # any kind of scaling done by barostat (if used). And the atoms are
    # harmonically restrained to the dummy atom. We have to redefine the
    # system, b/c we're adding new particles and this would clash with
    # modeller.topology.
    
    
    # Add the harmonic restraints on the positions
    # of specified atoms
    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint)
    nonbonded = [force for force in system.getForces()
                 if isinstance(force, NonbondedForce)][0]
    dummyIndex = []
    input_positions = PDBFile(min_pdb).getPositions()
    positions = input_positions
    # Go through the indices of all atoms that will be restrained
    for i in solute_heavy_atom_idx:
        j = system.addParticle(0)
        # ... and add a dummy/ghost atom next to it
        nonbonded.addParticle(0, 1, 0)
        # ... that won't interact with the restrained atom 
        nonbonded.addException(i, j, 0, 1, 0)
        # ... but will be have a harmonic restraint ('bond')
        # between the two atoms
        restraint.addBond(i, j, 0*unit.nanometers,
                          5*unit.kilocalories_per_mole/unit.angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])
        
    ## define integrator
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    ## define platform
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
    
    
    ## create simulation
    simulation = app.Simulation(top, system, integrator, platform, properties)
    simulation.context.setPositions(input_positions)
    
    ## run equilibration
    simulation.step(250000) ## run 500 ps of equilibration
    all_positions = simulation.context.getState(
        getPositions=True, enforcePeriodicBox = True).getPositions()
    
    # we don't want to write the dummy atoms, so we only
    # write the positions of atoms up to the first dummy atom index
    relevant_positions = all_positions[:dummyIndex[0]]
    out_file = os.path.join(out_dir,eq_pdb_name)
    PDBFile.writeFile(simulation.topology, relevant_positions,
                      open(out_file, 'w'),keepIds=True)
    
    return None



def production_bpmd_score(
    ref_sys_pdb, 
    ref_complex_pdb, 
    eql_pdb, 
    ff, 
    top, 
    out_dir, 
    idx, 
    md_pdb_name, 
    anchor_atom_idx, 
    lig_hvy_idx, 
    check_on_the_fly = False, 
    stop_criterion = 0.3,
    sim_type = 'plain', 
    run_time = 10, 
    time_step = 0.002, 
):
    """
    Production run with BPMD score
    
    Parameters
    ----------
    ref_sys_pdb: str
        Path to the reference system PDB file. used for the RMSD calculation.
    ref_complex_pdb: str
        Path to the reference complex PDB file. used for fitting the trajectory.
    eql_pdb: str
        Path to the equilibrated PDB file. used as the beginning of the production run.
    ff: openmm.app.ForceField
        Forcefield object.
    top: openmm.app.topology.Topology
        Topology object.
    pos: list
        List of positions.
    out_dir: str
        Output directory for the MD trajectory.
    idx: int
        current replica index.
    md_pdb_name: str
        Name of the MD PDB file.
    anchor_atom_idx: list
        List of the anchor atom indices.
    lig_hvy_idx: list
        List of the ligand heavy atom indices.
    check_on_the_fly: bool, optional
        If or not use the on-the-fly RMSD checking strategy, default is False.
    stop_criterion: float, optional
        The stop criterion for the metadynamics simulation, default is 0.2 nm.
    sim_type: str, optional
        Type of the simulation, default is 'plain'.
    run_time: int, optional
        Length of the simulation in nanoseconds.
    time_step: float, optional
        Time step for the simulation in picoseconds. Default is 0.002 ps.
    p_coupl: bool, optional
        If or not use pressure coupling, default is False.
    
    Returns 
    -------
    rmsd_score: float
        The RMSD score of the production run.
    mol_fail: bool
        If the molecule failed to pass the stop criterion (only useful when check_on_the_fly is True).
    """
    
    ## define outputs frequency and files
    steps_per_ns = int(1000/(time_step)) # number of steps per ns
    steps_per_ps = int(1/(time_step)) # number of steps per ps
    report_interval = int(steps_per_ns / 10) # report every 100 ps
    n_steps = int(run_time * 1000 / time_step)  # in steps
    trj_file = os.path.join(out_dir, f'rep_{idx}_md.xtc')
    log_file = os.path.join(out_dir, f'rep_{idx}_log.csv')
    
    ## define system to run metadynamics
    system = ff.createSystem(
        topology = top,
        nonbondedMethod=app.PME,
        constraints=HBonds,
        hydrogenMass=4*unit.amu,
        rigidWater = True
    )
    input_positions = PDBFile(eql_pdb).getPositions()
    
    ## add an 'empty' flat-bottom restraint to fix the issue with PBC.
    # Without one, RMSDForce object fails to account for PBC.
    k = 0*unit.kilojoules_per_mole  # NOTE - 0 kJ/mol constant
    upper_wall = 10.00*unit.nanometer
    fb_eq = '(k/2)*max(distance(g1,g2) - upper_wall, 0)^2'
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(lig_hvy_idx)
    upper_wall_rest.addGroup(anchor_atom_idx)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter('k', k)
    upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    system.addForce(upper_wall_rest)
    
    ## define the rmsd force
    rmsd_ref_positions = PDBFile(ref_sys_pdb).getPositions()
    alignment_indices = lig_hvy_idx + anchor_atom_idx
    rmsd = RMSDForce(rmsd_ref_positions, alignment_indices)
    
    ## set up the metadynamics
    grid_min, grid_max = 0.0, 1.0  # nm
    if sim_type == 'plain': # set gaussian height to 0 
        hill_height = 0.0 * unit.kilocalories_per_mole # kcal/mol
    if sim_type == 'bpmd':
        hill_height = 0.287 * unit.kilocalories_per_mole # kcal/mol
    hill_width = 0.002  # nm, also known as sigma
    grid_width = hill_width / 5
    n_grid = int(abs(grid_min - grid_max) / grid_width) # 'grid' here refers to the number of grid points
    rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width, periodic = False, gridWidth=n_grid)
    
    # define the metad object
    ## deposit bias every 1 ps, BF = 4, write bias every ns
    meta = Metadynamics(system, [rmsd_cv], 300.0*unit.kelvin, 4.0, hill_height,
                        steps_per_ps, biasDir=out_dir,
                        saveFrequency=report_interval)
    
    ## define integrator
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, time_step*unit.picoseconds)
    
    ## define platform
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
    
    ## create simulation
    simulation = app.Simulation(top, system, integrator, platform, properties)
    simulation.context.setPositions(input_positions)
    
    # define reporters
    simulation.reporters.append(XTCReporter(trj_file, report_interval, enforcePeriodicBox = True))  # every 100 ps
    simulation.reporters.append(StateDataReporter(
                                log_file, report_interval,
                                step=True, temperature=True, progress=True,
                                remainingTime=True, speed=True,
                                totalSteps=n_steps, separator=','))  # every 100 ps
    
    mol_fail = False
    
    ## run the metadynamics 500 steps at a time
    if check_on_the_fly:
        print(f"Running {sim_type} simulation with check on the fly.")
    else:
        print(f"Running {sim_type} simulation.")
        
    colvar_array = np.array([meta.getCollectiveVariables(simulation)])
    for i in range(0, int(n_steps), 500):
        if i % report_interval == 0:
            # log the stored COLVAR every 100ps
            np.save(os.path.join(out_dir,'COLVAR.npy'), colvar_array)
        meta.step(simulation, 500)
        current_cvs = meta.getCollectiveVariables(simulation)
        # record the CVs every 1 ps
        colvar_array = np.append(colvar_array, [current_cvs], axis=0)
        
        if check_on_the_fly:
            ## check the RMSD on the fly
            if (i != 0) and (i % steps_per_ns) == 0:
                print(f"Checking RMSD at {i/(steps_per_ns)} ns.")
                cur_rmsd_record = np.load(os.path.join(out_dir,'COLVAR.npy'))
                cur_interval_rmsd = np.mean(cur_rmsd_record[-int(steps_per_ns/500):])
                if cur_interval_rmsd > stop_criterion:
                    print(f"RMSD is too high at {cur_interval_rmsd} nm. Stopping the simulation at {i/(steps_per_ns)} ns.")
                    mol_fail = True
                    break
                else:
                    print(f"RMSD is {cur_interval_rmsd} nm at {i/(steps_per_ns)} ns, continuing the simulation.")
        
    # save the final CVs and fes
    np.save(os.path.join(out_dir,'COLVAR.npy'), colvar_array)
    grid_values = np.array([grid_min + i*grid_width for i in range(n_grid)])
    fes_values = meta.getFreeEnergy()
    fes = np.column_stack((grid_values, fes_values))
    np.save(os.path.join(out_dir,'fes.npy'), fes)
    
    
    ## save the final structure    
    final_positions = simulation.context.getState(
        getPositions=True, enforcePeriodicBox = True).getPositions()
    out_sys_pdb = os.path.join(out_dir, f"rep_{idx}_sys_{md_pdb_name}")
    out_complex_pdb = os.path.join(out_dir, f"rep_{idx}_complex_{md_pdb_name}")
    PDBFile.writeFile(simulation.topology, final_positions,open(out_sys_pdb, 'w'), keepIds=True)
    mda.Universe(out_sys_pdb).select_atoms('protein or resname UNK').write(out_complex_pdb)
    
    ## remove the solvent and ions in the trajectory and align to the reference complex
    u_md = mda.Universe(out_sys_pdb, trj_file)
    ref = mda.Universe(ref_complex_pdb)
    pro_lig = u_md.select_atoms('protein or resname UNK')
    u_md.trajectory.add_transformations(
        mda.transformations.center_in_box(pro_lig, center='mass'),  # Center on center of mass
        mda.transformations.wrap(u_md.atoms, compound='fragments'),
        mda.transformations.fit_rot_trans(pro_lig, ref, weights = "mass")
    )
    with mda.Writer(trj_file.replace('.xtc', '_pbc.xtc'), u_md.atoms.n_atoms) as W:
        for ts in u_md.trajectory:
            W.write(u_md.atoms)        
    with mda.Writer(os.path.join(out_dir, f"rep_{idx}_complex_md.xtc"), pro_lig.n_atoms) as W:
        for ts in u_md.trajectory:
            W.write(pro_lig.atoms)
            
    u_md.trajectory[-1]
    u_md.atoms.write(out_sys_pdb)
    u_md.select_atoms('protein or resname UNK').atoms.write(out_complex_pdb)
    u_md.select_atoms('protein').atoms.write(os.path.join(out_dir, f"rep_{idx}_pro.pdb"))
    u_md.select_atoms('resname UNK').atoms.write(os.path.join(out_dir, f"rep_{idx}_lig.pdb"))
    
    if sim_type == "plain": # for plain MD, the RMSD score is the mean rmsd over the trajectory
        rmsd_score = np.mean(colvar_array[:,0])
    if sim_type == "bpmd": # for BPMD, the RMSD score is the ensemble average after reweighting the metad
        rmsd_score = get_rmsd_score(os.path.join(out_dir,'fes.npy'))
        
    plot_trj_ifp(
        out_complex_pdb, 
        os.path.join(out_dir, f"rep_{idx}_complex_md.xtc"), 
        out_dir,
        f"rep_{idx}_ifp.png"
    )
    
    plot_trj_rmsd(
        ref_complex_pdb,
        os.path.join(out_dir, f"rep_{idx}_complex_md.xtc"),
        out_dir,
        f"rep_{idx}_rmsd.png"
    )
        
    trj_ifp_df = get_complex_prolif(
        out_complex_pdb, 
        os.path.join(out_dir, f"rep_{idx}_complex_md.xtc"), 
        os.path.join(out_dir, f"rep_{idx}_prolif.csv"),
        os.path.join(out_dir, f"ref_{idx}_prolif.pkl")
    )
    
    
    return rmsd_score, trj_ifp_df, mol_fail



def get_anchor_atoms(ref_pdb, ligand_name = 'UNK', scheme = 1):
    """
    Get the index of the anchor atoms.
    
    Parameters
    ----------
    ref_pdb: str
        Path to the reference complex PDB file.
    ligand_name: str
        Name of the ligand in the PDB file.
    scheme: int, optional
        Scheme to use for anchor atom selection.
    
    Returns
    -------
    anchor_atom_idx: list
        List of the anchor atom indices.
    
    """
    
    ## load the reference structure
    universe = mda.Universe(ref_pdb, format='XPDB', in_memory=True)
    
    ## if use the original scheme, use the C-alpha atom that minimizes the outward pointing vector as the single anchor atom
    if scheme == 1:
        mol = universe.select_atoms(f"resname {ligand_name}")
        pkt = universe.select_atoms(f"byres protein and around 6 resname {ligand_name}")
        # calculate the COG of mol and pkt
        mol_COM = mol.center_of_geometry()
        pkt_COM = mol.center_of_geometry()
        # calculate the outward vector
        outward_v = mol_COM - pkt_COM
        # get all the Calpha atoms around the ligand
        pkt_ca = universe.select_atoms(f"name CA and around 6 resname {ligand_name}")
        
        # loop over all these Cas and find the one that minimizes the projection of the outward vector
        best_ca = None
        max_projection = -np.inf
        for ca in pkt_ca:
            ca_v = mol_COM - ca.position
            projection = np.dot(ca_v, outward_v)
            if projection > max_projection:
                best_ca = ca
        anchor_atom_idx = [best_ca.id - 1]

    ## or use the alternative scheme as defined in the support info of the original BPMD paper
    else:
        ## ... finding the protein's COM ...
        #prot_com = universe.select_atoms('protein').center_of_mass()
        #x, y, z = prot_com[0], prot_com[1], prot_com[2]
        ## ... and taking the heavy backbone atoms within 10A of the COM
        #sel_str = f'point {x} {y} {z} 10 and backbone and not name H*'
        
        ## if no ref ligand is provided, use the Calpha atoms within 10A from the protein COM
        if len(universe.select_atoms(f"resname {ligand_name}").atoms) == 0:
            prot_com = universe.select_atoms('protein').center_of_mass()
            x, y, z = prot_com[0], prot_com[1], prot_com[2]
            anchor_atoms = universe.select_atoms(f'point {x} {y} {z} 10 and name CA')
            anchor_atom_idx = anchor_atoms.indices.tolist()
        ## or use all the CAs that's 6A from ligand as anchors
        else:
            anchor_atoms = universe.select_atoms(f"name CA and around 6 resname {ligand_name}")
            if len(anchor_atoms) == 0:
                raise ValueError('No Calpha atoms found within 6 ang of the center of mass of the protein. \
                        Check your input files.')
            anchor_atom_idx = anchor_atoms.indices.tolist()
    
    return anchor_atom_idx

def get_lig_hvy_atoms(ref_pdb, ligand_name = 'UNK'):
    """
    Get the indices of the heavy atoms of the ligand.
    
    Parameters
    ----------
    ref_pdb: str
        Path to the complex PDB file.
        
    ligand_name: str
        Name of the ligand in the PDB file.
        
    Returns
    -------
    lig_ha_idx: list
        List of the indices of the ligand heavy atoms.
    
    """
    
    # load the complex_structure
    universe = mda.Universe(ref_pdb, format='XPDB', in_memory=True)
    
    # Get indices of ligand heavy atoms
    lig = universe.select_atoms(f'resname {ligand_name} and not name H*')
    if len(lig) == 0:
        raise ValueError(f"Ligand with resname '{ligand_name}' not found.")

    lig_ha_idx = lig.indices.tolist()

    return lig_ha_idx


def get_rmsd_score(fes_profile):
    """
    To get the RMSD score (estimate) from the FES profile.
    
    Parameters
    ----------
    fes_profile: str
        The npy file of free energy surface profile as a function of rmsd.
    
    Returns
    -------
    rmsd_score: float
        The RMSD score.
    
    """
    rmsd_fes = np.load(fes_profile) # load the fes profile
    rmsd = rmsd_fes[:,0]
    fe = rmsd_fes[:,1]
    kt = 2.479 #kj/mol
    w = np.exp(-(fe)/kt)

    # return the rmsd estimate in nanometer
    return (np.dot(rmsd,w)/np.sum(w))


def get_ifp_score(ini_ifp_df, trj_ifp_df):
    """
    Calculating the IFP persistence score
    
    Parameters
    ----------
    ini_ifp_df  pd.DataFrame
        the initial IFP profile csv file (from prolif) to be compared with
    trj_ifp_df : pd.DataFrame
        the IFP profile frequency of the trajectory

    Returns 
    -------
    ifp_score: float
        The ifp persistence score
    """
    
    if len(ini_ifp_df) == 0 or len(trj_ifp_df) == 0:
        return 0.0
    
    ini_ifps = ini_ifp_df.sum().index
    n_frames = len(trj_ifp_df)
    
    persistence = 0
    for ifp, count in trj_ifp_df.sum().items():
        if ifp in ini_ifps:
            persistence += count / n_frames
            
    ifp_score = persistence / len(ini_ifps)
    
    return ifp_score 



def get_bpmd_score(rmsd_score, ifp_score, sim_type = 'plain'):
    """
    Get the BPMD score.
    
    Parameters
    ----------
    rmsd_score: float
        The RMSD score.
    ifp_score: float
        The IFP persistence score.
    sim_type: str, optional
        The simulation type. Default is 'plain'.
        
    Returns
    -------
    bpmd_score: float
        The BPMD score.
    
    """
    
    if sim_type == 'plain':
        d_w = 1.5
        i_w = -0.1
    elif sim_type == 'bpmd':
        d_w = 0.1
        i_w = -0.2
    else:
        raise ValueError(f"sim_type {sim_type} not supported. Only accept 'plain' and 'bpmd'")
    
    sigma_rmsd = 1 / (np.exp(d_w*(rmsd_score - 1.7)))
    rmsd_term = sigma_rmsd/rmsd_score

    sigma_ifp = 1/ (np.exp(i_w*(ifp_score - 0.3)))
    ifp_term = 2 * sigma_ifp * ifp_score

    return rmsd_term + ifp_term
    
    
    
def trj_cluster(working_dir, n_reps, lig_resname = 'UNK'):
    """
    Run a clustering analysis on the trajectory.
    
    Parameters
    ----------
    working_dir: str
        Working directory for the simulation.
    n_reps: int 
        Number of replicas to run.
    lig_resname: str, optional
        Name of the ligand in the PDB file.
        
    Returns
    -------
    None
    
    """
    
    # to append the trajectory of each replica together 
    complex_trj_file_list = []
    for i in range(n_reps):
        rep_id = i+1
        complex_trj_file_list.append(os.path.join(working_dir, f"rep_{rep_id}/rep_{rep_id}_complex_md.xtc"))
    sys_trj_file_list = []
    for i in range(n_reps):
        rep_id = i+1
        sys_trj_file_list.append(os.path.join(working_dir, f"rep_{rep_id}/rep_{rep_id}_md.xtc"))
        
    # load the trajectory
    complex_topol_file = os.path.join(working_dir, 'ini_complex.pdb')
    u_complex = mda.Universe(complex_topol_file, complex_trj_file_list)
    
    sys_topol_file = os.path.join(working_dir, 'ini_sys.pdb')
    u_sys = mda.Universe(sys_topol_file, sys_trj_file_list)
    
    
    # running the clustering on the complex trajectory
    cluster_collection = encore.cluster(u_complex, select = f'(name CA and protein and around 5 resname {lig_resname}) or (resname {lig_resname} and not name H*)',allow_collapsed_result=True)

    # get the centroid frame of each cluster 
    cluster_size = [len(cluster) for cluster in cluster_collection.clusters]
    cluster_centroids = [cluster.centroid for cluster in cluster_collection]
    rep_frame_idx = cluster_centroids[np.argmax(cluster_size)]
    
    # save the centroid frame of first(largest) cluster
    lig_atoms = u_complex.atoms.select_atoms(f"resname {lig_resname}")
    pro_atoms = u_complex.atoms.select_atoms(f"protein")
    with mda.Writer(os.path.join(working_dir, 'clu_rep_sys.pdb'), u_sys.atoms.n_atoms) as W:
        u_sys.trajectory[rep_frame_idx]
        W.write(u_sys.atoms)
    with mda.Writer(os.path.join(working_dir, 'clu_rep_complex.pdb'), u_complex.atoms.n_atoms) as W:
        u_complex.trajectory[rep_frame_idx]
        W.write(u_complex.atoms)
    with mda.Writer(os.path.join(working_dir, 'clu_rep_lig.pdb'), lig_atoms.n_atoms) as W:
        u_complex.trajectory[rep_frame_idx]
        W.write(lig_atoms)
    with mda.Writer(os.path.join(working_dir, 'clu_rep_pro.pdb'), pro_atoms.n_atoms) as W:
        u_complex.trajectory[rep_frame_idx]
        W.write(pro_atoms)

    try:
        clu_rep_prolif_df = get_pair_prolif(
            pro_pdb = os.path.join(working_dir, 'clu_rep_pro.pdb'),
            lig_pdb = os.path.join(working_dir, 'clu_rep_lig.pdb'),
            lig_sdf = os.path.join(working_dir, 'ligand.sdf'),
            df_out = os.path.join(working_dir, 'clu_rep_prolif.csv'),
            ifp_out = os.path.join(working_dir, 'clu_rep_prolif.pkl'),
            no_HY = True
        )
    except Exception as e:
        print(f"Error getting pair prolif: {e}")
        clu_rep_prolif_df = None
    
    
    return os.path.join(working_dir, 'clu_rep_complex.pdb')

def get_pair_prolif(pro_pdb, lig_pdb, lig_sdf = None, df_out = 'prolif.csv', ifp_out = 'prolif.pkl', no_HY = False):
    """
    Get the protein ligand fingerprint from the trajectory
    
    Parameters
    ----------
    pro_pdb: str
        The name of the protein PDB file.
    lig_pdb: str
        The name of the ligand PDB file.
    lig_sdf: str, optional
        The name of the ligand SDF file.
    df_out: str, optional
        The name of the output csv file.
    ifp_out: str, optional
        The name of the output pickle file.
    no_HY: bool, optional
        If True, drop hydrophobic and vdw.
    """
    
    #pro = Chem.MolFromPDBFile(pro_pdb, removeHs=False, sanitize=False)
    #lig = mda.Universe(lig_pdb).atoms.convert_to('RDKIT')
    
    lig = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
    if lig_sdf is not None:
        template = Chem.SDMolSupplier(lig_sdf, removeHs=False)[0]
        lig = AllChem.AssignBondOrdersFromTemplate(template, lig)
    lig_pdb_block = Chem.MolToPDBBlock(lig)

    u_pro = mda.Universe(pro_pdb).select_atoms('protein')
    u_lig = mda.Universe(io.StringIO(lig_pdb_block), format = 'PDB')
    u_complex = mda.Merge(u_pro.atoms, u_lig.atoms)
    u_pkt = u_complex.select_atoms(f"byres protein and around 6 resname UNK")
    u_pkt.atoms.guess_bonds()    
    
    pkt_mol = plf.Molecule.from_mda(u_pkt)
    lig_mol = plf.Molecule.from_rdkit(lig)
    
    fp_list = [
        'Anionic',
        'Cationic',
        
        'CationPi',
        'PiCation',
        
        'EdgeToFace',
        'FaceToFace',
        
        'HBAcceptor',
        'HBDonor',
        
        'Hydrophobic',
        'VdWContact',
        
        'XBAcceptor',
        'XBDonor'
    ]
    
    if no_HY:
        fp_list = [
            'Anionic',
            'Cationic',
            
            'CationPi',
            'PiCation',
            
            'EdgeToFace',
            'FaceToFace',
            
            'HBAcceptor',
            'HBDonor',
            
            'XBAcceptor',
            'XBDonor'
        ]
    
    fp = plf.Fingerprint(fp_list)
    fp.run_from_iterable([lig_mol],pkt_mol, n_jobs = 1, progress = False)
    
    fp_df = fp.to_dataframe()
    fp_df.to_csv(df_out)
    fp.to_pickle(ifp_out)
    
    return fp_df

def get_complex_prolif(top, trj = None, df_out = 'prolif.csv', ifp_out = 'prolif.pkl', no_HY = False, n_jobs = 1, progress = False):
    """
    Get the protein ligand fingerprint from the trajectory
    
    Parameters
    ----------
    top: str
        The name of the PDB file.
    trj: str, optional
        The name of the trajectory file.
    df_out: str, optional
        The name of the output csv file.
    ifp_out: str, optional
        The name of the output pickle file. 
    no_HY: str, optional
        If True, hydrophobic and vdw would not be consider
    n_jobs: int, optional
        The number of jobs to run in parallel.          
    """
    
    u = mda.Universe(top, trj) if not trj is None else mda.Universe(top)
    pro_sel = u.select_atoms('byres protein and around 6 resname UNK')
    lig_sel = u.select_atoms('resname UNK')
    lig_sel.guess_bonds()
    pro_sel.guess_bonds()
    
    fp_list = [
        'Anionic',
        'Cationic',
        
        'CationPi',
        'PiCation',
        
        'EdgeToFace',
        'FaceToFace',
        
        'HBAcceptor',
        'HBDonor',
        
        'Hydrophobic',
        'VdWContact',
        
        'XBAcceptor',
        'XBDonor'
    ]
    
    if no_HY:
        fp_list = [
            'Anionic',
            'Cationic',
            
            'CationPi',
            'PiCation',
            
            'EdgeToFace',
            'FaceToFace',
            
            'HBAcceptor',
            'HBDonor',
            
            'XBAcceptor',
            'XBDonor'
        ]
    
    fp = plf.Fingerprint(fp_list)
    fp.run(u.trajectory, lig_sel, pro_sel, n_jobs = n_jobs, progress = progress)
    
    ifp_df = fp.to_dataframe()
    ifp_df.to_csv(df_out)
    fp.to_pickle(ifp_out)
    
    return ifp_df


def plot_trj_ifp(top, trj, out_dir, ifp_plot_name):
    """
    Plot the protein ligand fingerprint from the trajectory
    
    Parameters
    ----------
    top: str
        The name of the PDB file.
    trj: str
        The name of the trajectory file.
    out_dir: str
        The name of the output directory.
    plot_image_name: str 
        The name of the output ifp frequency plot.
    """
    
    ifp_df = get_complex_prolif(top, trj, os.path.join(out_dir, 'trj_prolif_noHY.csv'), os.path.join(out_dir, 'trj_prolif_noHY.pkl'), no_HY = True)
    
    ifp_df = ifp_df.iloc[:, 1:]
    
    ifp_labels = []
    ifp_ratios = []
    
    for col in ifp_df.columns:
        ifp_name = f"{col[1]}_{col[2]}"
        ratio = (ifp_df[col] == True).sum() / ifp_df[col].notna().sum()
        if not col[2] in ['Hydrophobic', 'VdWContact']:
            ifp_labels.append(ifp_name)
            ifp_ratios.append(ratio)
    
    # Adaptively set width: 0.6 inch per bar, min 10, max 28 for better SaaS display
    import matplotlib.ticker as mticker

    n_bars = len(ifp_labels)
    width = min(max(10, n_bars * 0.6), 28)
    fig, ax = plt.subplots(figsize=(width, 7))

    # Use a modern color palette
    palette = sns.color_palette("Set2", n_colors=n_bars)
    bars = ax.bar(ifp_labels, ifp_ratios, color=palette, edgecolor='black', linewidth=0.8)

    # Add value labels on top of bars
    for bar, ratio in zip(bars, ifp_ratios):
        height = bar.get_height()
        ax.annotate(f"{ratio:.2f}", 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#333333')

    ax.set_xlabel('Interaction Type', fontsize=14, labelpad=12)
    ax.set_ylabel('Frequency', fontsize=14, labelpad=12)
    ax.set_title('Protein-Ligand Interaction Fingerprint Frequency', fontsize=16, pad=18, weight='bold')

    # Improve x-tick labels
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(ifp_labels, rotation=35, ha='right', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)

    # Remove top/right spines for a cleaner look
    sns.despine(ax=ax)

    # Add grid for y-axis
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=2)
    plt.savefig(os.path.join(out_dir, ifp_plot_name), dpi=180, bbox_inches='tight', transparent=True)
    plt.close()
    
def plot_trj_rmsd(top, trj, out_dir, rmsd_plot_name):
    """
    Plot the rmsd of the trajectory

    Parameters
    ----------
    top: str
        The name of the PDB file.
    trj: str
        The name of the trajectory file.
    out_dir: str
        The name of the output directory.
    rmsd_plot_name: str
        The name of the output rmsd plot.
    """
    
    u = mda.Universe(top, trj)
    lig_hv = u.select_atoms('resname UNK and not name H*')


    ref = mda.Universe(top, trj)
    ref.trajectory[0]
    ref_lig_hv = ref.select_atoms('resname UNK and not name H*')

    n_frames = len(u.trajectory)
    rmsd_lig_fit = np.zeros(n_frames)
    rmsd_pro_fit = np.zeros(n_frames)
    sim_time = np.zeros(n_frames)
    
    for ts in u.trajectory:
        align.alignto(u, ref, select = {'mobile': 'resname UNK and not name H*', 'reference': 'resname UNK and not name H*'})
        rmsd_lig_fit[ts.frame] = rms.rmsd(lig_hv.positions, ref_lig_hv.positions)
        sim_time[ts.frame] = ts.time/1000
    
    for ts in u.trajectory:
        align.alignto(u, ref, select = {'mobile': 'protein and not name H*', 'reference': 'protein and not name H*'})
        rmsd_pro_fit[ts.frame] = rms.rmsd(lig_hv.positions, ref_lig_hv.positions)


    
    rmsd_df = pd.DataFrame({
        "time (ns)": sim_time,
        "rmsd_lig_fit": rmsd_lig_fit,
        "rmsd_pro_fit": rmsd_pro_fit
    })
    rmsd_csv_path = os.path.join(out_dir, os.path.splitext(rmsd_plot_name)[0] + "_rmsd.csv")
    rmsd_df.to_csv(rmsd_csv_path, index=False)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    # Plot with thicker lines and markers
    ax.plot(
        sim_time, rmsd_lig_fit, 
        label="Ligand RMSD (fit to itself)", 
        color="#2563eb", 
        linewidth=2.2, 
        marker='o', 
        markersize=4, 
        alpha=0.85
    )
    ax.plot(
        sim_time, rmsd_pro_fit, 
        label="Ligand RMSD (after protein fit)", 
        color="#f59e42", 
        linewidth=2.2, 
        marker='s', 
        markersize=4, 
        alpha=0.85
    )

    # Modern, clean style
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # Axis labels and title with improved font
    ax.set_xlabel("Time (ns)", fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel("Ligand Heavy Atom RMSD (Å)", fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title("Ligand Heavy Atom RMSD Over Trajectory", fontsize=17, fontweight='bold', pad=15)

    # Ticks and grid
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune='lower'))
    ax.grid(True, linestyle=':', linewidth=1, color='#cbd5e1', alpha=0.7, zorder=0)

    # Remove top/right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Legend with better placement and style
    legend = ax.legend(
        fontsize=10, 
        loc='upper left', 
        frameon=True, 
        framealpha=0.75, 
        borderpad=0.8, 
        edgecolor='#cbd5e1'
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor('#f1f5f9')

    fig.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, rmsd_plot_name), dpi=220, bbox_inches='tight', transparent=True)
    plt.close()
    
    
    
def run_bpmd_udf(sdf: str, pdb: str, n_replica: int = 3, md_time = 2):
    """
    function for running BPMD simulation in a udf
    
    Parameters
    ----------
    sdf: str
        sdf block of the ligand
    pdb: str
        ref complex pdb address on s3
    n_replica: int, optional
        number of replicas to run, default is 3
    md_time: int, optional
        length of the MD simulation in nanoseconds, default is 10
        
    Returns
    -------
    The result of the simulation (rmsd scores)
    
    """
    os.makedirs('/code/bpmd_folder', exist_ok = True)
    rmsds, label = run_simulation(lig_sdf_block = sdf, ref_complex_pdb = pdb, working_dir = '/code/bpmd_folder', ref_lig_name = 'MOL', sim_type = 'plain', check_on_the_fly = False, n_reps = n_replica, md_time = md_time)
    return_list = [rmsd for rmsd in rmsds]
    return_list.append(label)
    return return_list


#def minimize_res_lig_pair()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run a md simulaiton')
    parser.add_argument('--sdf', type=str, help='sdf', default='test/test.sdf')
    parser.add_argument(f"--pdb", type=str, help='complex pdb', default='test/test.pdb')
    parser.add_argument(f"--lig_name", type=str, help='ligand name', default='UNK')
    parser.add_argument(f"--dir", type=str, help="working directory", default='test')
    parser.add_argument(f"--sim_type", type=str, help="type of simulation", default='plain')
    parser.add_argument(f"--stop_criterion", type=float, help="stop criterion", default=0.3)
    args = parser.parse_args()
    
    sdf_addr = args.sdf
    pdb_addr = args.pdb
    lig_name = args.lig_name
    work_dir = args.dir
    sim_type = args.sim_type
    stop_rmsd = args.stop_criterion
    
    os.makedirs(work_dir, exist_ok = True)
    
    sdf_supplier = Chem.SDMolSupplier(sdf_addr, removeHs=False)
    if not sdf_supplier:
        raise ValueError(f"Could not read SDF file: {sdf_addr}")
    sdf_block = Chem.MolToMolBlock(sdf_supplier[0])
    
    rmsds, label = run_simulation(
        sdf_block, 
        pdb_addr, 
        working_dir = work_dir, 
        ref_lig_name = lig_name, 
        n_reps = 1, 
        md_time = 10,
        bpmd_score = True,
        sim_type = sim_type, 
        check_on_the_fly = True, 
        stop_criterion = stop_rmsd, 
    )
        
    print(f"Simulation finished. RMSD scores: {rmsds}, label: {label}")

