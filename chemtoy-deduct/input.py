# Data sources
database(
    thermoLibraries = ['primaryThermoLibrary'],
    reactionLibraries = [],
    seedMechanisms = [],
    kineticsDepositories = ['training'],
    kineticsFamilies = 'default',
    kineticsEstimator = 'rate rules',
)

# List of species
# ---- Hydrogen / Oxygen Core ----
species(label='H2',  reactive=True, structure=SMILES('[H][H]'))
species(label='H',   reactive=True, structure=SMILES('[H]'))
species(label='O2',  reactive=True, structure=SMILES('[O][O]'))
species(label='O',   reactive=True, structure=SMILES('[O]'))
species(label='OH',  reactive=True, structure=SMILES('[OH]'))
species(label='HO2', reactive=True, structure=SMILES('[O]O'))
species(label='H2O', reactive=True, structure=SMILES('O'))

# ---- Nitrogen ----
species(label='N2', reactive=False, structure=SMILES('N#N'))
species(label='N',  reactive=True,  structure=SMILES('[N]'))
species(label='NH', reactive=True,  structure=SMILES('[NH]'))
species(label='NH2', reactive=True, structure=SMILES('[NH2]'))
species(label='NH3', reactive=True, structure=SMILES('N'))

#species(label='NO',  reactive=True, structure=SMILES('[N]=O'))
#
## Correct radical representation of NO2
#species(label='NO2', reactive=True, structure=SMILES('[O]N=O'))
#
## ---- Sulfur ----
#species(label='S',   reactive=True, structure=SMILES('[S]'))
#species(label='HS',  reactive=True, structure=SMILES('[SH]'))
#species(label='H2S', reactive=True, structure=SMILES('S'))
#
#species(label='SO',  reactive=True, structure=SMILES('[S]=O'))
#species(label='SO2', reactive=True, structure=SMILES('O=S=O'))
#
## ---- Halogens ----
#species(label='Cl',  reactive=True, structure=SMILES('[Cl]'))
#species(label='Cl2', reactive=True, structure=SMILES('ClCl'))
#species(label='HCl', reactive=True, structure=SMILES('Cl'))
#
#species(label='F',   reactive=True, structure=SMILES('[F]'))
#species(label='F2',  reactive=True, structure=SMILES('F[F]')
#)
#species(label='HF',  reactive=True, structure=SMILES('F'))# Reaction systems

simpleReactor(
    temperature=(1350,'K'),
    pressure=(1.0,'bar'),
    initialMoleFractions={
        "H2": 0.1,
        "O2": 0.1,
        "N2": 0.1,
        "NH3": 0.05,
        #"SO2": 0.05,
        #"Cl2": 0.05,
        #"F2": 0.05,
    },
    terminationTime=(1e1,'s'),
)

simulator(
    atol=1e-16,
    rtol=1e-8,
)

model(
    toleranceKeepInEdge=0.0,
    toleranceMoveToCore=0.1,
    toleranceInterruptSimulation=0.1,
    maximumEdgeSpecies=100000,
    filterReactions=True,
)

options(
    units='si',
    generateOutputHTML=False,
    generatePlots=False,
    saveEdgeSpecies=True,
    saveSimulationProfiles=True,
)

