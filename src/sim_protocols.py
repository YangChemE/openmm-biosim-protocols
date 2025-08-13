from process import *

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



def plain_complex_md_protocol(
    protein_pdb: str,
    ligand_sdf: str,
    working_dir: str,
    n_reps: int = 1,
    md_time: int = 10,
    solvation: bool = True,
    eql_posre: bool = True,
    p_coupl: bool = True,
    use_gpu: bool = True,
):
    
    
    ## create the working directory 
    os.makedirs(working_dir, exist_ok=True)
    
    # Set up logging to a file
    log_file = os.path.join(working_dir, 'simulation.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.info(f"Working directory created: {working_dir}")
    
    
    ## define the output filenames 
    # output file names for minimization and equilibration 
    ini_pdb_name = 'ini.pdb'
    ini_complex_pdb_name = 'ini_complex.pdb'
    ini_pro_pdb_name = 'ini_pro.pdb'
    ini_lig_pdb_name = 'ini_lig.pdb'
    min_pdb_name = 'min.pdb'
    min_complex_pdb_name = 'min_complex.pdb'
    min_pro_pdb_name = 'min_pro.pdb'
    min_lig_pdb_name = 'min_lig.pdb'
    nvt_pdb_name = 'nvt.pdb'
    nvt_complex_pdb_name = 'nvt_complex.pdb'
    nvt_pro_pdb_name = 'nvt_pro.pdb'
    nvt_lig_pdb_name = 'nvt_lig.pdb'
    npt_pdb_name = 'npt.pdb'
    npt_complex_pdb_name = 'npt_complex.pdb'
    npt_pro_pdb_name = 'npt_pro.pdb'
    npt_lig_pdb_name = 'npt_lig.pdb'
    
    # output file names for production 
    md_pdb_name = 'md.pdb'
    md_complex_pdb_name = 'md_complex.pdb'
    md_lig_pdb_name = 'md_lig.pdb'
    md_pro_pdb_name = 'md_pro.pdb'
    md_trj_name = 'md_trj.xtc'
    md_complex_trj_name = 'md_complex_trj.xtc'
    md_log_name = 'md_log.csv'
    
    # output file names for prolif analysis 
    min_prolif_csv_name = 'min_prolif.csv'
    min_prolif_pkl_name = 'min_prolif.pkl'
    md_prolif_csv_name = 'md_prolif.csv'
    md_trj_prolif_csv_name = 'md_trj_prolif.csv'
    md_prolif_pkl_name = 'md_prolif.pkl'
    md_trj_ifp_plot_name = 'md_trj_ifp_plot.png'
    md_trj_rmsd_plot_name = 'md_trj_rmsd_plot.png'

    
    ## prepare the input files
    logger.info(f"Preparing the input files...")
    logger.info(f"Prepare the input protein...")
    remove_protein_hydrogens(protein_pdb, os.path.join(working_dir, 'protein_noH.pdb'))
    prepared_protein = prepare_protein(os.path.join(working_dir, 'protein_noH.pdb'), ignore_missing_residues=False, ignore_terminal_missing_residues = True)
    PDBFile.writeFile(prepared_protein.topology, prepared_protein.positions, open(os.path.join(working_dir, ini_pro_pdb_name), 'w'),keepIds=True)
    
    logger.info(f"Generate the complex pdb...")
    u_pro = mda.Universe(os.path.join(working_dir, ini_pro_pdb_name))
    rdkit_lig = Chem.SDMolSupplier(ligand_sdf, removeHs = False)[0]
    omm_lig = rdkit_to_openmm(rdkit_lig, name = 'UNK')
    complex_top, complex_pos, _ = merge_protein_and_ligand(prepared_protein, omm_lig)
    PDBFile.writeFile(complex_top, complex_pos, open(os.path.join(working_dir, ini_complex_pdb_name), 'w'),keepIds=True)
    
    
    ## generating the forcefield 
    logger.info(f"Generating the forcefield...")
    forcefield = app.ForceField('amber14-all.xml')
    if solvation:
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    off_mol = Molecule.from_rdkit(rdkit_lig, allow_undefined_stereo = True)
    try:
            off_mol.assign_partial_charges(partial_charge_method="am1bcc")
    except:
        logger.info(f"sqm can't be run properly, using mmff94 charge instead off am1bcc.")
        off_mol.assign_partial_charges(partial_charge_method="mmff94")
    gaff = GAFFTemplateGenerator(molecules = off_mol)
    forcefield.registerTemplateGenerator(gaff.generator)
    
    # use modeller to help solvate the system and get a resonable box 
    modeller = app.Modeller(complex_top, complex_pos)
    modeller.addSolvent(forcefield, padding = 1.2*unit.nanometers, ionicStrength = 0.15*unit.molar, boxShape = 'dodecahedron')
    complex_top.setPeriodicBoxVectors(modeller.topology.getPeriodicBoxVectors())
    if solvation:
        sys_top, sys_pos = modeller.getTopology(), modeller.getPositions()
    else:
        sys_top, sys_pos = complex_top, complex_pos
    PDBFile.writeFile(sys_top, sys_pos, open(os.path.join(working_dir, ini_pdb_name), 'w'),keepIds=True)
    u_ini = mda.Universe(os.path.join(working_dir, ini_pdb_name))
    u_ini.select_atoms('protein').atoms.write(os.path.join(working_dir, ini_pro_pdb_name))
    u_ini.select_atoms('resname UNK').atoms.write(os.path.join(working_dir, ini_lig_pdb_name))
    
    
    ## create the system and simulation 
    logger.info(f"Creating the openmmm system and simulation...")
    system = forcefield.createSystem(
        topology = sys_top,
        nonbondedMethod = app.PME,
        nonbondedCutoff = 1.0*unit.nanometers,
        constraints = app.HBonds,
        hydrogenMass = 4*unit.amu,
        rigidWater = True
    )
    
    # define the platform 
    if use_gpu:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
    else:
        platform = Platform.getPlatformByName('CPU')
        properties = {}
    
    # define the integrator 
    integrator = LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        0.002*unit.picoseconds
    )
    
    # create the simulation 
    simulation = app.Simulation(sys_top, system, integrator, platform, properties)
    simulation.context.setPositions(sys_pos)
    
    
    ## run minimization
    logger.info(f"Running minimization...")
    logger.info(f"number of particles in the system: {simulation.context.getSystem().getNumParticles()}")
    simulation.minimizeEnergy()
    min_topology = simulation.topology 
    min_state = simulation.context.getState(
        positions = True,
        velocities = True,
        forces = True, 
        energy = True,
        parameterDerivatives = True,
        integratorParameters = True,
        enforcePeriodicBox = True
    )
    PDBFile.writeFile(min_topology, min_state.getPositions(), open(os.path.join(working_dir, min_pdb_name), 'w'),keepIds=True)
    u_min = mda.Universe(os.path.join(working_dir, min_pdb_name))
    u_min.select_atoms('protein or resname UNK').write(os.path.join(working_dir, min_complex_pdb_name))
    u_min.select_atoms('protein').atoms.write(os.path.join(working_dir, min_pro_pdb_name))
    u_min.select_atoms('resname UNK').atoms.write(os.path.join(working_dir, min_lig_pdb_name))
    
    ## run nvt equilibration 
    logger.info(f"Running NVT equilibration...")
    if eql_posre:
        logger.info(f"Using position restraints on protein heavy atoms for NVT equilibration...")
        protein_heavy_atoms = u_min.select_atoms('protein and not name H*')
        pro_heavy_atom_idx = [int(idx) for idx in protein_heavy_atoms.indices]
        logger.info(f"Number of protein heavy atoms: {len(pro_heavy_atom_idx)}")
        
        # Add the harmonic restraints on the positions of specified atoms
        restraint = HarmonicBondForce()
        restraint.setUsesPeriodicBoundaryConditions(True)
        system.addForce(restraint)
        nonbonded = [force for force in system.getForces()
                    if isinstance(force, NonbondedForce)][0]
        dummyIndex = []
        input_positions = PDBFile(os.path.join(working_dir, min_pdb_name)).getPositions()
        positions = input_positions
        # Go through the indices of all atoms that will be restrained
        for i in pro_heavy_atom_idx:
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
        
    nvt_integrator = LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        0.002*unit.picoseconds
    )
    nvt_simulation = app.Simulation(min_topology, system, nvt_integrator, platform, properties)
    nvt_simulation.context.setPositions(input_positions)
    
    # run the nvt equilibration
    logger.info(f"Number of particles in the system after adding virtual atoms for position restraints: {nvt_simulation.context.getSystem().getNumParticles()}")
    nvt_simulation.step(250000) ## run 500 ps of equilibration
    nvt_topology = nvt_simulation.topology
    nvt_state = nvt_simulation.context.getState(
        positions = True,
        velocities = True,
        forces = True, 
        energy = True,
        parameterDerivatives = True,
        integratorParameters = True,
        enforcePeriodicBox = True
    )
    nvt_all_positions = nvt_state.getPositions()
    nvt_all_velocities = nvt_state.getVelocities()
    # we don't want to write the dummy atoms, so we only
    # write the positions of atoms up to the first dummy atom index
    nvt_relevant_positions = nvt_all_positions[:dummyIndex[0]] if eql_posre else nvt_all_positions
    nvt_relevant_velocities = nvt_all_velocities[:dummyIndex[0]] if eql_posre else nvt_all_velocities
    
    ## run another nvt equilibration without posre
    system = forcefield.createSystem(
        topology = sys_top,
        nonbondedMethod = app.PME,
        nonbondedCutoff = 1.0*unit.nanometers,
        constraints = app.HBonds,
        hydrogenMass = 4*unit.amu,
        rigidWater = True
    )   
    nvt_integrator = LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        0.002*unit.picoseconds
    )
    nvt2_simulation = app.Simulation(min_topology, system, nvt_integrator, platform, properties)
    nvt2_simulation.context.setPositions(nvt_relevant_positions)
    nvt2_simulation.context.setVelocities(nvt_relevant_velocities)
    nvt2_simulation.step(250000) ## run 500 ps of equilibration
    nvt2_topology = nvt2_simulation.topology
    nvt2_state = nvt2_simulation.context.getState(
        positions = True,
        velocities = True,
        forces = True, 
        energy = True,
        parameterDerivatives = True,
        integratorParameters = True,
        enforcePeriodicBox = True
    )
    
    PDBFile.writeFile(nvt2_topology, nvt2_state.getPositions(), open(os.path.join(working_dir, nvt_pdb_name), 'w'),keepIds=True)
    u_nvt = mda.Universe(os.path.join(working_dir, nvt_pdb_name))
    u_nvt.select_atoms('protein or resname UNK').write(os.path.join(working_dir, nvt_complex_pdb_name))
    u_nvt.select_atoms('protein').atoms.write(os.path.join(working_dir, nvt_pro_pdb_name))
    u_nvt.select_atoms('resname UNK').atoms.write(os.path.join(working_dir, nvt_lig_pdb_name))
    
    
    ## run npt equilibration 
    if p_coupl:
        logger.info(f"Running NPT equilibration with pressure coupling...")
        system.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
        npt_integrator = LangevinIntegrator(
            300*unit.kelvin,
            1.0/unit.picoseconds,
            0.002*unit.picoseconds
        )
        npt_simulation = app.Simulation(nvt2_topology, system, npt_integrator, platform, properties)
        npt_simulation.context.setState(nvt2_state)
        logger.info(f"Number of particles in the system: {npt_simulation.context.getSystem().getNumParticles()}")
        npt_simulation.step(250000) ## run 500 ps of equilibration
        npt_topology = npt_simulation.topology
        npt_state = npt_simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox = True)
        npt_all_positions = npt_state.getPositions()
        npt_all_velocities = npt_state.getVelocities()
        PDBFile.writeFile(npt_topology, npt_all_positions, open(os.path.join(working_dir, npt_pdb_name), 'w'),keepIds=True)
        u_npt = mda.Universe(os.path.join(working_dir, npt_pdb_name))
        u_npt.select_atoms('protein or resname UNK').write(os.path.join(working_dir, npt_complex_pdb_name))
        u_npt.select_atoms('protein').atoms.write(os.path.join(working_dir, npt_pro_pdb_name))
        u_npt.select_atoms('resname UNK').atoms.write(os.path.join(working_dir, npt_lig_pdb_name))
    
    
    ## define the eql state
    if p_coupl:
        eql_state = npt_state
        eql_topology = npt_topology
    else:
        eql_state = nvt_state
        eql_topology = nvt_topology
    
    
    ## run production simulation
    # define outputs frequency and files
    time_step = 0.002
    steps_per_ns = int(1000/(time_step)) # number of steps per ns
    steps_per_ps = int(1/(time_step)) # number of steps per ps
    report_interval = int(steps_per_ns / 10) # report every 100 ps
    n_steps = int(md_time * 1000 / time_step)  # in steps

    for idx in range(n_reps):
        rep_id = idx + 1
        rep_dir = os.path.join(working_dir, f'rep_{rep_id}')
        os.makedirs(rep_dir, exist_ok=True)
        
        # define the output files
        sys_pdb = os.path.join(rep_dir, f'{md_pdb_name}')
        complex_pdb = os.path.join(rep_dir, f'{md_complex_pdb_name}')
        pro_pdb = os.path.join(rep_dir, f'{md_pro_pdb_name}')
        lig_pdb = os.path.join(rep_dir, f'{md_lig_pdb_name}')
        trj_file = os.path.join(rep_dir, f'{md_trj_name}')
        complex_trj_file = os.path.join(rep_dir, f'{md_complex_trj_name}')
        log_file = os.path.join(rep_dir, f'{md_log_name}')
        
        # Initialize the simulation
        logger.info(f"Running the rep {rep_id} production simulation...")
        md_integrator = LangevinIntegrator(
            300*unit.kelvin,
            1.0/unit.picoseconds,
            0.002*unit.picoseconds
        )
        md_simulation = app.Simulation(eql_topology, system, md_integrator, platform, properties)
        md_simulation.context.setState(eql_state)
        md_simulation.context.setStepCount(0)
        
        # define reporters
        md_simulation.reporters.append(XTCReporter(trj_file, report_interval, enforcePeriodicBox = True))  # every 100 ps
        md_simulation.reporters.append(StateDataReporter(
                                    log_file, report_interval,
                                    step=True, temperature=True, progress=True,
                                    remainingTime=True, speed=True,
                                    totalSteps=n_steps, separator=','))  # every 100 ps
        
        # run the production simulation
        logger.info(f"number of particles in the system: {md_simulation.context.getSystem().getNumParticles()}")
        md_simulation.step(n_steps)
        logger.info(f"Finished the rep {rep_id} production simulation.")
        
        # record the final structure
        logger.info(f"Processing the trajectory of the rep {rep_id} ...")
        final_topology = md_simulation.topology
        final_state = md_simulation.context.getState(getPositions=True, enforcePeriodicBox = True)
        final_positions = final_state.getPositions()
        PDBFile.writeFile(final_topology, final_positions, open(sys_pdb, 'w'),keepIds=True)
        
        # handling the PBC for visulization
        u_md = mda.Universe(sys_pdb, trj_file)
        ref = mda.Universe(os.path.join(working_dir, min_complex_pdb_name))
        pro_lig = u_md.select_atoms('protein or resname UNK')
        u_md.trajectory.add_transformations(
            mda.transformations.center_in_box(pro_lig, center='mass'),  # Center on center of mass
            mda.transformations.wrap(u_md.atoms, compound='fragments'),
            mda.transformations.fit_rot_trans(pro_lig, ref, weights = "mass")
        )
        with mda.Writer(trj_file.replace('.xtc', '_pbc.xtc'), u_md.atoms.n_atoms) as W:
            for ts in u_md.trajectory:
                W.write(u_md.atoms)
        with mda.Writer(complex_trj_file, pro_lig.n_atoms) as W:
            for ts in u_md.trajectory:
                W.write(pro_lig.atoms)
        # write the final stucture
        u_md.trajectory[-1]
        u_md.atoms.write(sys_pdb)
        u_md.select_atoms('protein or resname UNK').atoms.write(complex_pdb)
        u_md.select_atoms('protein').atoms.write(pro_pdb)
        u_md.select_atoms('resname UNK').atoms.write(lig_pdb)
        
        logger.info(f"Finished processing the trajectory of the rep {rep_id}. Making the ifp and rmsd plot.")
        # make the ifp freq plot
        plot_trj_ifp(complex_pdb, complex_trj_file, rep_dir, md_trj_ifp_plot_name)
        
        # make rmsd plot
        plot_trj_rmsd(os.path.join(working_dir, ini_complex_pdb_name), complex_trj_file, rep_dir, md_trj_rmsd_plot_name)

        logger.info(f"Rep {rep_id} done!")


def complex_md_bpmd_score_protocol(
    lig_sdf_block: str, 
    ref_complex_pdb: str, 
    working_dir: str, 
    ref_lig_name: str, 
    n_reps: int = 3, 
    md_time: int = 10,
    use_gpu: bool = True,
    sim_type: str = 'plain', 
    check_on_the_fly: bool = False, 
):
    """
    Run a molecular dynamics simulation. 
    Note that due to the need of 
    calculating rmsd (RMSDForce) during the simulation, this protocol 
    only support nvt ensemble, as implemented in the original BPMD and 
    OpenBPMD.

    Parameters
    ----------
    lig_sdf_block: str
        SDF block containing the ligand.
    ref_complex_pdb: str
        PDB file containing the reference complex. Be aware that this ref pdb means \
        the reference complex. i.e. the cocrystral structure with active ligand.
    working_dir: str
        Working directory for the simulation.
    ref_lig_name: str
        Name of the ligand in the PDB file.
    n_reps: int, optional
        Number of replicas to run.
    md_time: int, optional
        Length of the MD simulation in nanoseconds.
    use_gpu: bool, optional
        If or not to use GPU.
    sim_type: str, optional
        Type of simulation to run. Options are 'plain' and 'bpmd'.
    check_on_the_fly: bool, optional
        If or not to check the RMSD on the fly.
    
    Returns
    -------
    result: str
        Path to the result PDB file.
    """

    
    
    ## create the working directory 
    os.makedirs(working_dir, exist_ok=True)
    
    # Set up logging to a file
    log_file = os.path.join(working_dir, 'simulation.log')
    logging.basicConfig(   
        filename=log_file,
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.info(f"Working directory created: {working_dir}")
    
    if sim_type == 'plain':
        stop_criterion = 0.3
    if sim_type == 'bpmd':
        stop_criterion = 0.6
    
    
    logger.info(
        """
        Prepare the simulation\n
        1. Change the ligand resname to MOL\n
        2. Remove protein hydrogens\n
        3. Fix the protein with pdbFixer\n
        4. Get the anchor atom index\n
        5. Prepare the ligand\n
        6. Convert the ligand to OpenMM format\n
        7. Merge protein and ligand\n
        8. Run the simulation\n
        ...\n
        """
    )
    ## change the lig resname to MOL
    os.system(f"sed -i 's/{ref_lig_name}/MOL/g' {ref_complex_pdb}")
    ## remove protein hydrogens 
    ref_complex_pdb_noH = os.path.join(working_dir, 'ref_complex_noH.pdb')
    remove_protein_hydrogens(ref_complex_pdb, ref_complex_pdb_noH)
    
    ## prepare protein and write it out to a PDB file
    prepared_protein = prepare_protein(ref_complex_pdb_noH, ignore_missing_residues=False, ignore_terminal_missing_residues = True)
    PDBFile.writeFile(prepared_protein.topology, prepared_protein.positions, open(os.path.join(working_dir, 'ref_pro_prepared.pdb'), 'w'),keepIds=True)
    
    ## get the ref complex pdb with prepared protein
    os.system(f'grep ATOM {os.path.join(working_dir, "ref_pro_prepared.pdb")} > {os.path.join(working_dir, "ref_pro_block.pdb")}')
    os.system(f"grep MOL {ref_complex_pdb} > {os.path.join(working_dir, 'ref_lig.pdb')}")
    os.system(f"cat {os.path.join(working_dir, 'ref_pro_block.pdb')} {os.path.join(working_dir, 'ref_lig.pdb')} > {os.path.join(working_dir, 'ref_complex_prepared.pdb')}")
    
    ## get the anchor atom index based on the fixed ref complex pdb
    ref_complex_pdb_prepared = os.path.join(working_dir, 'ref_complex_prepared.pdb')
    anchor_idx = get_anchor_atoms(ref_complex_pdb_prepared, ligand_name = 'MOL', scheme = 2)
    logger.info(f"Anchor atom index: {anchor_idx}, {type(anchor_idx)}")
    
    ## get the ligand, assuming the ligand has correct/desired bond orders
    # and wrtie it out to an sdf file
    rdkit_ligand = Chem.MolFromMolBlock(lig_sdf_block, removeHs=False)
    Chem.SDWriter(os.path.join(working_dir, 'ligand.sdf')).write(rdkit_ligand)
    
    ## convert the ligand to OpenMM format
    omm_ligand = rdkit_to_openmm(rdkit_ligand, name = 'UNK')
    
    ## merge protein and ligand
    ini_complex_pdb = os.path.join(working_dir, 'ini_complex.pdb')
    complex_topology, complex_positions, md_complex_topology = merge_protein_and_ligand(prepared_protein, omm_ligand)
    PDBFile.writeFile(complex_topology, complex_positions, open(ini_complex_pdb, 'w'),keepIds=True)
    
    ## get the ligand heavy atom indices from the initial structure
    lig_hvy_idx = get_lig_hvy_atoms(os.path.join(working_dir, 'ini_complex.pdb'))
    logger.info(f"Ligand heavy atom indices: {lig_hvy_idx}, {type(lig_hvy_idx)}")
    
    ## print out the total number of atoms without solvation
    logger.info(f"System has {complex_topology.getNumAtoms()} atoms.")
    
    ## generate forcefield
    forcefield, forcefield_vac = generate_forcefield(rdkit_ligand)
    
    ## solvate the system and save the sturcture
    modeller = app.Modeller(complex_topology, complex_positions)
    modeller.addSolvent(forcefield, padding = 1.2 * unit.nanometers, ionicStrength=0.15 * unit.molar, boxShape = 'dodecahedron')
    logger.info(f"The box size is {modeller.topology.getPeriodicBoxVectors()}")
    complex_topology.setPeriodicBoxVectors(modeller.topology.getPeriodicBoxVectors()) ## set the periodic box vectors for non-solvent topolgy
    ini_sys_pdb = os.path.join(working_dir, 'ini_sys.pdb')
    PDBFile.writeFile(modeller.topology, modeller.positions, open(ini_sys_pdb, 'w'),keepIds=True)
    logger.info(f"System has {modeller.topology.getNumAtoms()} atoms after solvation.")
   
    # Define platform properties
    if use_gpu:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaDeviceIndex': '0','CudaPrecision': 'mixed'}
    else:
        platform = Platform.getPlatformByName('CPU')
        properties = {}
    
    
    ## energy minimization and saving pbc dealt complex structure
    logger.info(f"Running minimization...")
    min_sys_pdb = os.path.join(working_dir, 'min_sys.pdb')
    min_complex_pdb = os.path.join(working_dir, 'min_complex.pdb')
    minimization(forcefield, modeller.topology, modeller.positions, working_dir, 'min_sys.pdb')
    u_min = mda.Universe(min_sys_pdb)
    u_min.select_atoms('protein or resname UNK').write(min_complex_pdb)
    u_min.select_atoms('protein').write(os.path.join(working_dir, 'min_pro.pdb'))
    u_min.select_atoms('resname UNK').write(os.path.join(working_dir, 'min_lig.pdb'))
    try:
        min_ifp_df = get_pair_prolif(
            pro_pdb = os.path.join(working_dir, 'min_pro.pdb'), 
            lig_pdb = os.path.join(working_dir, 'min_lig.pdb'), 
            lig_sdf = os.path.join(working_dir, 'ligand.sdf'), 
            df_out = os.path.join(working_dir, 'min_prolif.csv'), 
            ifp_out = os.path.join(working_dir, 'min_prolif.pkl')
        )
    except Exception as e:
        logger.info(f"Error getting pair prolif: {e}")
        min_prolif_df = None
    
    ## equilibration and saving pbc dealt complex 
    logger.info(f"Running NVT equilibration...")
    eql_sys_pdb = os.path.join(working_dir, 'eql_sys.pdb')
    eql_complex_pdb = os.path.join(working_dir, 'eql_complex.pdb')
    nvt_equilibration(min_sys_pdb, forcefield, modeller.topology, working_dir, 'eql_sys.pdb')
    u_eql = mda.Universe(eql_sys_pdb)
    u_eql.select_atoms('protein or resname UNK').write(eql_complex_pdb)
    u_eql.select_atoms('protein').write(os.path.join(working_dir, 'eql_pro.pdb'))
    u_eql.select_atoms('resname UNK').write(os.path.join(working_dir, 'eql_lig.pdb'))
    

    rmsd_scores, ifp_scores, bpmd_scores = [], [], []
    ## run Nreps of plain MD
    for idx in range(0, n_reps):
        rep_idx = idx + 1
        rep_dir = os.path.join(working_dir,f'rep_{rep_idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)
        
        logger.info(f"Running {sim_type} simulation for rep {rep_idx}...")
        
        rmsd_score, trj_ifp_df, mol_fail = production_bpmd_score(
            eql_sys_pdb, 
            min_complex_pdb, 
            eql_sys_pdb, 
            forcefield, 
            modeller.topology, 
            rep_dir, 
            rep_idx, 
            'md.pdb', 
            anchor_idx, 
            lig_hvy_idx, 
            check_on_the_fly = check_on_the_fly, 
            stop_criterion = stop_criterion,
            sim_type = sim_type, 
            run_time = md_time
        )
        
        ifp_score = get_ifp_score(min_ifp_df, trj_ifp_df)
        bpmd_score = get_bpmd_score(rmsd_score, ifp_score, sim_type)
        
        
        if not mol_fail:
            rmsd_scores.append(rmsd_score)
            ifp_scores.append(ifp_score)
            bpmd_scores.append(bpmd_score)
            logger.info(f"Rep {rep_idx} done! RMSD score: {rmsd_score:.6f}, IFP score: {ifp_score:.6f}, BPMD score: {bpmd_score:.6f}")    
        else:
            logger.info(f"Rep {rep_idx} failed!")
            break
    
    
    # if early termination criterion met during check-on-the-fly, fill the rmsd record with 10000
    while len(rmsd_scores) < n_reps:
        rmsd_scores.append(10000)
        ifp_scores.append(0)
        bpmd_scores.append(0)
    
    rmsd_scores_array = np.array(rmsd_scores)
    ifp_scores_array = np.array(ifp_scores)
    bpmd_scores_array = np.array(bpmd_scores)
    np.save(os.path.join(working_dir,'rmsd_scores.npy'), rmsd_scores_array)
    np.save(os.path.join(working_dir, 'ifp_scores.npy'), ifp_scores_array)
    np.save(os.path.join(working_dir, 'bpmd_scores.npy'), bpmd_scores_array)
    
    # if early terminated, molecule considered failed and label as fail, no further analysis would be done.
    if mol_fail:
        label = 'fail'
    # if all reps done, molecule considered passed and label as pass, 
    # do further processing, clustering, vacumm minimization, and bpmd score calculation
    else:
        label = 'pass'
        clu_rep_complex_pdb = trj_cluster(working_dir, n_reps)
        vac_system = forcefield_vac.createSystem(
            complex_topology,
            nonbondedMethod=app.PME,
            constraints=HBonds,
            hydrogenMass=4*unit.amu
        )
        
        integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        vac_simulation = app.Simulation(complex_topology, vac_system, integrator, platform, properties)
        vac_simulation.context.setPositions(PDBFile(clu_rep_complex_pdb).getPositions())
        vac_simulation.minimizeEnergy()
        positions = vac_simulation.context.getState(getPositions=True, enforcePeriodicBox = True).getPositions()
        out_file = os.path.join(working_dir,'clu_rep_complex_minimized.pdb')
        PDBFile.writeFile(complex_topology, positions,
                        open(out_file, 'w'),keepIds=True)
        

        
        
    return rmsd_scores, ifp_scores, bpmd_scores, label

    
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex_pdb', type=str, required=True)
    parser.add_argument('--ligand_sdf', type=str, required=True)
    parser.add_argument('--working_dir', type=str, required=True)
    parser.add_argument('--n_reps', type=int, required=True)
    parser.add_argument('--md_time', type=int, required=True)
    args = parser.parse_args()
    
    
    """
    ## the plain md protocol with in npt
    plain_complex_md_protocol(
        protein_pdb = args.protein_pdb,
        ligand_sdf = args.ligand_sdf,
        working_dir = args.working_dir,
        n_reps = args.n_reps,
        md_time = args.md_time,
        solvation = True,
        eql_posre = True,
        p_coupl = True,
        use_gpu = False
    )
    """
    
    ## BPMD protocol in nvt
    rmsd_scores, ifp_scores, bpmd_scores, labels = complex_md_bpmd_score_protocol(
        lig_sdf_block = Chem.MolToMolBlock(Chem.SDMolSupplier('args.ligand_sdf', removeHs = False)[0]), 
        ref_complex_pdb = args.protein_pdb, 
        working_dir = args.working_dir, 
        ref_lig_name = 'UNL', 
        n_reps = args.n_reps, 
        md_time = args.md_time,
        use_gpu = True,
        sim_type = 'plain', 
        check_on_the_fly = False, 
        stop_criterion = 0.3, 
    )
    
        
        
        
        
        
        

    
    
    
    
    
