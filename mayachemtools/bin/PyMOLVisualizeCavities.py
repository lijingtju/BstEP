#!/bin/env python
#
# File: PyMOLVisualizeCavities.py
# Author: Manish Sud <msud@san.rr.com>
#
# Copyright (C) 2020 Manish Sud. All rights reserved.
#
# The functionality available in this script is implemented using PyMOL, a
# molecular visualization system on an open source foundation originally
# developed by Warren DeLano.
#
# This file is part of MayaChemTools.
#
# MayaChemTools is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# MayaChemTools is distributed in the hope that it will be useful, but without
# any warranty; without even the implied warranty of merchantability of fitness
# for a particular purpose.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MayaChemTools; if not, see <http://www.gnu.org/licenses/> or
# write to the Free Software Foundation Inc., 59 Temple Place, Suite 330,
# Boston, MA, 02111-1307, USA.
#

from __future__ import print_function

# Add local python path to the global path and import standard library modules...
import os
import sys;  sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), "..", "lib", "Python"))
import time
import re

# PyMOL imports...
try:
    import pymol
    # Finish launching PyMOL in  a command line mode for batch processing (-c)
    # along with the following options:  disable loading of pymolrc and plugins (-k);
    # suppress start up messages (-q)
    pymol.finish_launching(['pymol', '-ckq'])
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import PyMOL module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your PyMOL environment and try again.\n\n")
    sys.exit(1)

# MayaChemTools imports...
try:
    from docopt import docopt
    import MiscUtil
    import PyMOLUtil
except ImportError as ErrMsg:
    sys.stderr.write("\nFailed to import MayaChemTools module/package: %s\n" % ErrMsg)
    sys.stderr.write("Check/update your MayaChemTools environment and try again.\n\n")
    sys.exit(1)

ScriptName = os.path.basename(sys.argv[0])
Options = {}
OptionsInfo = {}

def main():
    """Start execution of the script"""
    
    MiscUtil.PrintInfo("\n%s (PyMOL v%s; %s) Starting...\n" % (ScriptName, pymol.cmd.get_version()[0], time.asctime()))
    
    (WallClockTime, ProcessorTime) = MiscUtil.GetWallClockAndProcessorTime()
    
    # Retrieve command line arguments and options...
    RetrieveOptions()
    
    # Process and validate command line arguments and options...
    ProcessOptions()

    # Perform actions required by the script...
    GenerateCavitiesVisualization()
    
    MiscUtil.PrintInfo("\n%s: Done...\n" % ScriptName)
    MiscUtil.PrintInfo("Total time: %s" % MiscUtil.GetFormattedElapsedTime(WallClockTime, ProcessorTime))

def GenerateCavitiesVisualization():
    """Generate visualization for cavities."""

    Outfile = OptionsInfo["PMLOutfile"]
    OutFH = open(Outfile, "w")
    if OutFH is None:
        MiscUtil.PrintError("Failed to open output fie %s " % Outfile)
    
    MiscUtil.PrintInfo("\nGenerating file %s..." % Outfile)

    # Setup header...
    WritePMLHeader(OutFH, ScriptName)
    WritePyMOLParameters(OutFH)
    
    # Load reffile for alignment..
    if OptionsInfo["Align"]:
        WriteAlignReference(OutFH)

    # Setup view for each input file...
    FirstComplex = True
    FirstComplexFirstChainName = None
    for FileIndex in range(0, len(OptionsInfo["InfilesInfo"]["InfilesNames"])):
        # Setup PyMOL object names...
        PyMOLObjectNames = SetupPyMOLObjectNames(FileIndex)

        # Setup complex view...
        WriteComplexView(OutFH, FileIndex, PyMOLObjectNames, FirstComplex)
        
        SpecifiedChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]
        FirstChain = True
        for ChainID in SpecifiedChainsAndLigandsInfo["ChainIDs"]:
            if FirstComplex and FirstChain:
                FirstComplexFirstChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]
                
            WriteChainView(OutFH, FileIndex, PyMOLObjectNames, ChainID)
            
            # Setup ligand views...
            FirstLigand = True
            for LigandID in SpecifiedChainsAndLigandsInfo["LigandIDs"][ChainID]:
                WriteChainLigandView(OutFH, FileIndex, PyMOLObjectNames, ChainID, LigandID)
                
                # Set up ligand level group...
                Enable, Action = [False, "close"]
                if FirstLigand:
                    FirstLigand = False
                    Enable, Action = [True, "open"]
                GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Ligands"][ChainID][LigandID]["ChainLigandGroup"], PyMOLObjectNames["Ligands"][ChainID][LigandID]["ChainLigandGroupMembers"], Enable, Action)
            
            # Setup Chain level group...
            Enable, Action = [False, "close"]
            if FirstChain:
                FirstChain = False
                Enable, Action = [True, "open"]
            GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID]["ChainGroup"], PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"], Enable, Action)
    
        # Set up complex level group...
        Enable, Action = [False, "close"]
        if FirstComplex:
            FirstComplex = False
            Enable, Action = [True, "open"]
        GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["PDBGroup"], PyMOLObjectNames["PDBGroupMembers"], Enable, Action)
        
        # Delete empty PyMOL objects...
        DeleteEmptyPyMOLObjects(OutFH, FileIndex, PyMOLObjectNames)
        
    if OptionsInfo["Align"]:
        DeleteAlignReference(OutFH)

    if FirstComplexFirstChainName is not None:
        OutFH.write("""\ncmd.orient("%s", animate = -1)\n""" % FirstComplexFirstChainName)
    else:
        OutFH.write("""\ncmd.orient("visible", animate = -1)\n""")
    
    OutFH.close()

    # Generate PSE file as needed...
    if OptionsInfo["PSEOut"]:
        GeneratePyMOLSessionFile()

def WritePMLHeader(OutFH, ScriptName):
    """Write out PML setting up complex view"""

    HeaderInfo = PyMOLUtil.SetupPMLHeaderInfo(ScriptName)
    OutFH.write("%s\n" % HeaderInfo)

def WritePyMOLParameters(OutFH):
    """Write out PyMOL global parameters. """

    PMLCmds = []
    PMLCmds.append("""cmd.set("transparency", %.2f, "", 0)""" % (OptionsInfo["SurfaceTransparency"]))
    PMLCmds.append("""cmd.set("label_font_id", %s)""" % (OptionsInfo["LabelFontID"]))

    PMLCmds.append("""cmd.set("cavity_cull", %.1f, "", 0)""" % (OptionsInfo["CavityCullSize"]))
    PMLCmds.append("""cmd.set("surface_cavity_radius", -%.1f, "", 0)""" % (OptionsInfo["CavityRadius"]))
    PMLCmds.append("""cmd.set("surface_cavity_cutoff", -%.1f, "", 0)""" % (OptionsInfo["CavityCutoff"]))
    
    PML = "\n".join(PMLCmds)
    
    OutFH.write("""\n""\n"Setting up PyMOL gobal parameters..."\n""\n""")
    OutFH.write("%s\n" % PML)
    
def WriteAlignReference(OutFH):
    """Setup object for alignment reference """

    RefFileInfo = OptionsInfo["RefFileInfo"]
    RefFile = RefFileInfo["RefFileName"]
    RefName = RefFileInfo["PyMOLObjectName"]
    
    PMLCmds = []
    PMLCmds.append("""cmd.load("%s", "%s")""" % (RefFile, RefName))
    PMLCmds.append("""cmd.hide("everything", "%s")""" % (RefName))
    PMLCmds.append("""cmd.disable("%s")""" % (RefName))
    PML = "\n".join(PMLCmds)
    
    OutFH.write("""\n""\n"Loading %s and setting up view for align reference..."\n""\n""" % RefFile)
    OutFH.write("%s\n" % PML)
    
def WriteAlignComplex(OutFH, FileIndex, PyMOLObjectNames):
    """Setup alignment of complex to reference"""

    RefFileInfo = OptionsInfo["RefFileInfo"]
    RefName = RefFileInfo["PyMOLObjectName"]
    
    ComplexName = PyMOLObjectNames["Complex"]
    
    if re.match("^FirstChain$", OptionsInfo["AlignMode"], re.I):
        RefFirstChainID = RefFileInfo["ChainsAndLigandsInfo"]["ChainIDs"][0]
        RefAlignSelection = "%s and chain %s" % (RefName, RefFirstChainID)
        
        ComplexFirstChainID = RetrieveFirstChainID(FileIndex)
        ComplexAlignSelection = "%s and chain %s" % (ComplexName, ComplexFirstChainID)
    else:
        RefAlignSelection = RefName
        ComplexAlignSelection = ComplexName

    PML = PyMOLUtil.SetupPMLForAlignment(OptionsInfo["AlignMethod"], RefAlignSelection, ComplexAlignSelection)
    OutFH.write("""\n""\n"Aligning %s against reference %s ..."\n""\n""" % (ComplexAlignSelection, RefAlignSelection))
    OutFH.write("%s\n" % PML)
    
def DeleteAlignReference(OutFH):
    """Delete alignment reference object."""
    
    RefName = OptionsInfo["RefFileInfo"]["PyMOLObjectName"]
    OutFH.write("""\n""\n"Deleting alignment reference object %s..."\n""\n""" % RefName)
    OutFH.write("""cmd.delete("%s")\n""" % RefName)

def WriteComplexView(OutFH, FileIndex, PyMOLObjectNames, FirstComplex):
    """Write out PML for viewing polymer complex."""

    # Setup complex...
    Infile = OptionsInfo["InfilesInfo"]["InfilesNames"][FileIndex]
    PML = PyMOLUtil.SetupPMLForPolymerComplexView(PyMOLObjectNames["Complex"], Infile, True)
    OutFH.write("""\n""\n"Loading %s and setting up view for complex..."\n""\n""" % Infile)
    OutFH.write("%s\n" % PML)

    if OptionsInfo["Align"]:
        # No need to align complex on to itself...
        if not (re.match("^FirstInputFile$", OptionsInfo["AlignRefFile"], re.I) and FirstComplex):
            WriteAlignComplex(OutFH, FileIndex, PyMOLObjectNames)
    
    if OptionsInfo["SurfaceComplex"]:
        # Setup hydrophobic surface...
        PML = PyMOLUtil.SetupPMLForHydrophobicSurfaceView(PyMOLObjectNames["ComplexHydrophobicSurface"], PyMOLObjectNames["Complex"], ColorPalette = OptionsInfo["SurfaceColorPalette"], Enable = False)
        OutFH.write("\n%s\n" % PML)
    
    # Setup complex group...
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["ComplexGroup"], PyMOLObjectNames["ComplexGroupMembers"], False, "close")

def WriteChainView(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write out PML for viewing chain."""
    
    OutFH.write("""\n""\n"Setting up views for chain %s..."\n""\n""" % ChainID)
    
    ChainComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]
    
    # Setup chain complex group view...
    WriteChainComplexViews(OutFH, FileIndex, PyMOLObjectNames, ChainID)

    # Setup chain view...
    WriteChainAloneViews(OutFH, FileIndex, PyMOLObjectNames, ChainID)
    
    # Setup chain solvent view...
    PML = PyMOLUtil.SetupPMLForSolventView(PyMOLObjectNames["Chains"][ChainID]["Solvent"], ChainComplexName, False)
    OutFH.write("\n%s\n" % PML)

    # Setup chain inorganic view...
    PML = PyMOLUtil.SetupPMLForInorganicView(PyMOLObjectNames["Chains"][ChainID]["Inorganic"], ChainComplexName, False)
    OutFH.write("\n%s\n" % PML)

def WriteChainComplexViews(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write chain complex views. """
    
    # Setup chain complex...
    ChainComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]
    PML = PyMOLUtil.SetupPMLForPolymerChainComplexView(ChainComplexName, PyMOLObjectNames["Complex"], ChainID, True)
    OutFH.write("%s\n" % PML)

    if OptionsInfo["SurfaceChainComplex"]:
        # Setup hydrophobic surface...
        PML = PyMOLUtil.SetupPMLForHydrophobicSurfaceView(PyMOLObjectNames["Chains"][ChainID]["ChainComplexHydrophobicSurface"], ChainComplexName, ColorPalette = OptionsInfo["SurfaceColorPalette"], Enable = False)
        OutFH.write("\n%s\n" % PML)
    
    # Setup chain complex group...
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroup"], PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"], False, "close")
    
def WriteChainAloneViews(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Write individual chain views. """

    ChainComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]
    
    # Setup chain view...
    ChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]
    PML = PyMOLUtil.SetupPMLForPolymerChainView(ChainName, ChainComplexName, True)
    OutFH.write("\n%s\n" % PML)

    WriteChainAloneResidueTypesView(OutFH,  FileIndex, PyMOLObjectNames, ChainID)
    WriteChainAloneCavitiesAndSurfacesView(OutFH,  FileIndex, PyMOLObjectNames, ChainID)

    # Setup chain group...
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroup"], PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"], True, "open")

def WriteChainLigandView(OutFH, FileIndex, PyMOLObjectNames, ChainID, LigandID):
    """Write out PML for viewing ligand in a chain."""

    GroupID = "Ligand"
    ComplexName = PyMOLObjectNames["Chains"][ChainID]["ChainComplex"]
    LigandName = PyMOLObjectNames["Ligands"][ChainID][LigandID]["Ligand"]
    
    # Setup main object...
    GroupTypeObjectID = "%s" % (GroupID)
    GroupTypeObjectName = PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupTypeObjectID]
    
    OutFH.write("""\n""\n"Setting up views for ligand %s in chain %s..."\n""\n""" % (LigandID, ChainID))
    PML = PyMOLUtil.SetupPMLForLigandView(GroupTypeObjectName, ComplexName, LigandID, True)
    OutFH.write("%s\n" % PML)
    
    # Setup ball and stick view...
    BallAndStickNameID = "%sBallAndStick" % (GroupID)
    BallAndStickName = PyMOLObjectNames["Ligands"][ChainID][LigandID][BallAndStickNameID]
    PML = PyMOLUtil.SetupPMLForBallAndStickView(BallAndStickName, GroupTypeObjectName, Enable = False)
    OutFH.write("\n%s\n" % PML)
            
    # Setup group....
    GroupNameID = "%sGroup" % (GroupID)
    GroupMembersID = "%sGroupMembers" % (GroupID)
    GroupName = PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupNameID]
    GroupMembers = PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupMembersID]

    Action = "close"
    Enable = True
    GenerateAndWritePMLForGroup(OutFH, GroupName, GroupMembers, Enable, Action)

def WriteChainAloneCavitiesAndSurfacesView(OutFH,  FileIndex, PyMOLObjectNames, ChainID):
    """Write out PML for viewing cavities and pockets in chains. """

    if not GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
        return
    
    ChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]
    
    for SubGroupType in ["Cavities", "Surface"]:
        SubGroupID = "ChainAlone%sGroup" % (SubGroupType)
        SubGroupMembersID = "%sMembers" % (SubGroupID)
        
        ProcessingCavities = True if re.match("^Cavities$", SubGroupType, re.I) else False
        
        # Turn off lines display for cavity surfaces...
        DisplayStyle = None if ProcessingCavities else "cartoon"
        
        # Setup a generic color surface...
        SurfaceID = "%sSurface" % (SubGroupID)
        SurfaceName = PyMOLObjectNames["Chains"][ChainID][SurfaceID]
        ColorName =  OptionsInfo["SurfaceCavityColor"] if ProcessingCavities else OptionsInfo["SurfaceColor"]
        EnableStatus =  True if ProcessingCavities else False
        PML = PyMOLUtil.SetupPMLForSurfaceView(SurfaceName, ChainName,  Enable = EnableStatus, Color = ColorName)
        OutFH.write("\n%s\n" % PML)
        
        if ProcessingCavities:
            OutFH.write("""cmd.set("surface_cavity_mode", %d, "%s")\n""" % (OptionsInfo["SurfaceCavityMode"], SurfaceName))
        
        if GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
            # Setup surface colored by hydrophobicity...
            HydrophobicSurfaceID = "%sHydrophobicSurface" % (SubGroupID)
            HydrophobicSurfaceName = PyMOLObjectNames["Chains"][ChainID][HydrophobicSurfaceID]
            PML = PyMOLUtil.SetupPMLForHydrophobicSurfaceView(HydrophobicSurfaceName, ChainName, ColorPalette = OptionsInfo["SurfaceColorPalette"], Enable = False)
            OutFH.write("\n%s\n" % PML)
            
            if ProcessingCavities:
                OutFH.write("""cmd.set("surface_cavity_mode", %d, "%s")\n""" % (OptionsInfo["SurfaceCavityMode"], HydrophobicSurfaceName))
            
            # Setup surface colored by hyrdophobicity and charge...
            HydrophobicChargeSurfaceID = "%sHydrophobicChargeSurface" % (SubGroupID)
            HydrophobicChargeSurfaceName = PyMOLObjectNames["Chains"][ChainID][HydrophobicChargeSurfaceID]
            PML = PyMOLUtil.SetupPMLForHydrophobicAndChargeSurfaceView(HydrophobicChargeSurfaceName, ChainName, OptionsInfo["AtomTypesColorNames"]["HydrophobicAtomsColor"], OptionsInfo["AtomTypesColorNames"]["NegativelyChargedAtomsColor"], OptionsInfo["AtomTypesColorNames"]["PositivelyChargedAtomsColor"], OptionsInfo["AtomTypesColorNames"]["OtherAtomsColor"], Enable = False, DisplayAs = None)
            OutFH.write("\n%s\n" % PML)
            
            if ProcessingCavities:
                OutFH.write("""cmd.set("surface_cavity_mode", %d, "%s")\n""" % (OptionsInfo["SurfaceCavityMode"], HydrophobicChargeSurfaceName))
        
        if GetChainAloneSurfaceChainElectrostaticsStatus(FileIndex, ChainID):
            # Setup electrostatics surface...
            ElectrostaticsGroupID = "%sElectrostaticsGroup" % (SubGroupID)
            ElectrostaticsGroupMembersID = "%sElectrostaticsGroupMembers" % (SubGroupID)
            ElectrostaticsGroupName = PyMOLObjectNames["Chains"][ChainID][ElectrostaticsGroupID]
            ElectrostaticsGroupMembers = PyMOLObjectNames["Chains"][ChainID][ElectrostaticsGroupMembersID]
            WriteSurfaceElectrostaticsView(SubGroupType, OutFH, ChainName, ElectrostaticsGroupName, ElectrostaticsGroupMembers, DisplayAs = DisplayStyle, SurfaceCavityMode = OptionsInfo["SurfaceCavityMode"])

        # Setup surface group...
        GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID][SubGroupID], PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID], True, "open")

def WriteChainAloneResidueTypesView(OutFH,  FileIndex, PyMOLObjectNames, ChainID):
    """Write out PML for viewing residue types for a chain. """

    if not GetChainAloneResidueTypesStatus(FileIndex, ChainID):
        return
    
    ChainName = PyMOLObjectNames["Chains"][ChainID]["ChainAlone"]
    
    # Setup residue types objects...
    ResiduesGroupIDPrefix = "ChainAloneResidues"
    for SubGroupType in ["Aromatic", "Hydrophobic", "Polar", "Positively_Charged", "Negatively_Charged", "Other"]:
        SubGroupID = re.sub("_", "", SubGroupType)

        ResiduesObjectID = "%s%sResidues" % (ResiduesGroupIDPrefix, SubGroupID)
        ResiduesObjectName = PyMOLObjectNames["Chains"][ChainID][ResiduesObjectID]

        ResiduesSurfaceObjectID = "%s%sSurface" % (ResiduesGroupIDPrefix, SubGroupID)
        ResiduesSurfaceObjectName = PyMOLObjectNames["Chains"][ChainID][ResiduesSurfaceObjectID]

        ResiduesColor = OptionsInfo["ResidueTypesParams"][SubGroupType]["Color"] 
        ResiduesNames = OptionsInfo["ResidueTypesParams"][SubGroupType]["Residues"]

        NegateResidueNames = True if re.match("^Other$", SubGroupType, re.I) else False
        WriteResidueTypesResiduesAndSurfaceView(OutFH, ChainName, ResiduesObjectName, ResiduesSurfaceObjectName, ResiduesColor, ResiduesNames, NegateResidueNames)

        # Setup sub groups for residue types..
        ResiduesSubGroupID = "%s%sGroup" % (ResiduesGroupIDPrefix, SubGroupID)
        ResiduesSubGroupMembersID = "%s%sGroupMembers" % (ResiduesGroupIDPrefix, SubGroupID)

        # Setup residue type sub groups...
        GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID][ResiduesSubGroupID], PyMOLObjectNames["Chains"][ChainID][ResiduesSubGroupMembersID], True, "close")
        
    # Setup residue types group...
    GenerateAndWritePMLForGroup(OutFH, PyMOLObjectNames["Chains"][ChainID]["ChainAloneResiduesGroup"], PyMOLObjectNames["Chains"][ChainID]["ChainAloneResiduesGroupMembers"], False, "close")

def WriteResidueTypesResiduesAndSurfaceView(OutFH, SelectionObjectName, Name, SurfaceName, ResiduesColor, ResiduesNames, NegateResidueNames):
    """Write residue types residues and surface view. """

    ResidueNamesSelection = "+".join(ResiduesNames)
    if NegateResidueNames:
        Selection = "%s and (not resn %s)" % (SelectionObjectName, ResidueNamesSelection)
    else:
        Selection = "%s and (resn %s)" % (SelectionObjectName, ResidueNamesSelection)

    # Setup residues...
    PML = PyMOLUtil.SetupPMLForSelectionDisplayView(Name, Selection, "lines", ResiduesColor, True)
    OutFH.write("\n%s\n" % PML)

    # Setup surface...
    PML = PyMOLUtil.SetupPMLForSelectionDisplayView(SurfaceName, Selection, "surface", ResiduesColor, True)
    OutFH.write("\n%s\n" % PML)
    
def WriteSurfaceElectrostaticsView(Mode, OutFH, SelectionObjectName, ElectrostaticsGroupName, ElectrostaticsGroupMembers, DisplayAs = None, SurfaceCavityMode = 2):
    """Write out PML for viewing surface electrostatics. """

    if len(ElectrostaticsGroupMembers) == 5:
        Name, ContactPotentialName, MapName, LegendName, VolumeName = ElectrostaticsGroupMembers
    else:
        Name, ContactPotentialName, MapName, LegendName = ElectrostaticsGroupMembers
        VolumeName = None

    PMLCmds = []

    # Setup chain...
    PMLCmds.append("""cmd.create("%s", "(%s)")""" % (Name, SelectionObjectName))

    # Setup vacuum electrostatics surface along with associated objects...
    PMLCmds.append("""util.protein_vacuum_esp("%s", mode=2, quiet=0, _self=cmd)""" % (Name))
    PMLCmds.append("""cmd.set_name("%s_e_chg", "%s")""" % (Name, ContactPotentialName))
    
    if DisplayAs is not None:
        PMLCmds.append("""cmd.show("%s", "(%s)")""" % (DisplayAs, ContactPotentialName))

    if re.match("^Cavities$", Mode, re.I):
        if SurfaceCavityMode is not None:
            PMLCmds.append("""cmd.set("surface_cavity_mode", %d, "%s")\n""" % (SurfaceCavityMode, ContactPotentialName))
        
    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(ContactPotentialName, Enable = True))
    
    PMLCmds.append("""cmd.set_name("%s_e_map", "%s")""" % (Name, MapName))
    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(MapName, Enable = False))
    
    PMLCmds.append("""cmd.set_name("%s_e_pot", "%s")""" % (Name, LegendName))
    PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(LegendName, Enable = False))

    if VolumeName is not None:
        PMLCmds.append("""cmd.volume("%s", "%s", "%s", "(%s)")""" % (VolumeName, MapName, "esp", Name))
        PMLCmds.append(PyMOLUtil.SetupPMLForEnableDisable(VolumeName, Enable = False))
    
    # Delete name and take it out from the group membership. It is
    # is already part of ContactPotential object.
    PMLCmds.append("""cmd.delete("%s")""" % (Name))
    ElectrostaticsGroupMembers.pop(0)
    
    PML = "\n".join(PMLCmds)
    
    OutFH.write("\n%s\n" % PML)
    
    # Setup group...
    GenerateAndWritePMLForGroup(OutFH, ElectrostaticsGroupName, ElectrostaticsGroupMembers, False, "close")

def GenerateAndWritePMLForGroup(OutFH, GroupName, GroupMembers, Enable = False, Action = "close"):
    """Generate and write PML for group. """
    
    PML = PyMOLUtil.SetupPMLForGroup(GroupName, GroupMembers, Enable, Action)
    OutFH.write("""\n""\n"Setting up group %s..."\n""\n""" % GroupName)
    OutFH.write("%s\n" % PML)

def GeneratePyMOLSessionFile():
    """Generate PME file from PML file. """

    PSEOutfile = OptionsInfo["PSEOutfile"]
    PMLOutfile = OptionsInfo["PMLOutfile"]
    
    MiscUtil.PrintInfo("\nGenerating file %s..." % PSEOutfile)
    
    PyMOLUtil.ConvertPMLFileToPSEFile(PMLOutfile, PSEOutfile)
    
    if not os.path.exists(PSEOutfile):
        MiscUtil.PrintWarning("Failed to generate PSE file, %s..." % (PSEOutfile))
    
    if not OptionsInfo["PMLOut"]:
        MiscUtil.PrintInfo("Deleting file %s..." % PMLOutfile)
        os.remove(PMLOutfile)

def DeleteEmptyPyMOLObjects(OutFH, FileIndex, PyMOLObjectNames):
    """Delete empty PyMOL objects. """
    
    if OptionsInfo["AllowEmptyObjects"]:
        return
    
    SpecifiedChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]
    for ChainID in SpecifiedChainsAndLigandsInfo["ChainIDs"]:
        OutFH.write("""\n""\n"Checking and deleting empty objects for chain %s..."\n""\n""" % (ChainID))
        
        # Delete any chain level objects...
        WritePMLToCheckAndDeleteEmptyObjects(OutFH, PyMOLObjectNames["Chains"][ChainID]["Solvent"])
        WritePMLToCheckAndDeleteEmptyObjects(OutFH, PyMOLObjectNames["Chains"][ChainID]["Inorganic"])

        # Delete residue type objects...
        DeleteEmptyChainResidueTypesObjects(OutFH, FileIndex, PyMOLObjectNames, ChainID)

def DeleteEmptyChainResidueTypesObjects(OutFH, FileIndex, PyMOLObjectNames, ChainID):
    """Delete empty chain residue objects. """
    
    if not GetChainAloneResidueTypesStatus(FileIndex, ChainID):
        return
    
    ResiduesGroupIDPrefix = "ChainAloneResidues"
    for GroupType in ["Aromatic", "Hydrophobic", "Polar", "Positively_Charged", "Negatively_Charged", "Other"]:
        GroupID = re.sub("_", "", GroupType)
        
        ResiduesGroupID = "%s%sGroup" % (ResiduesGroupIDPrefix, GroupID)
        GroupName = PyMOLObjectNames["Chains"][ChainID][ResiduesGroupID]
        
        GroupObjectNamesList = []
        
        ResiduesObjectID = "%s%sResidues" % (ResiduesGroupIDPrefix, GroupID)
        ResiduesObjectName = PyMOLObjectNames["Chains"][ChainID][ResiduesObjectID]
        GroupObjectNamesList.append(ResiduesObjectName)
        
        ResiduesSurfaceObjectID = "%s%sSurface" % (ResiduesGroupIDPrefix, GroupID)
        ResiduesSurfaceObjectName = PyMOLObjectNames["Chains"][ChainID][ResiduesSurfaceObjectID]
        GroupObjectNamesList.append(ResiduesSurfaceObjectName)
        
        GroupObjectNames = ",".join(GroupObjectNamesList)
        WritePMLToCheckAndDeleteEmptyObjects(OutFH, GroupObjectNames, GroupName)

def WritePMLToCheckAndDeleteEmptyObjects(OutFH, ObjectName, ParentObjectName = None):
    """Write PML to check and delete empty PyMOL objects. """
    
    if ParentObjectName is None:
        PML = """CheckAndDeleteEmptyObjects("%s")""" % (ObjectName)
    else:
        PML = """CheckAndDeleteEmptyObjects("%s", "%s")""" % (ObjectName, ParentObjectName)
    
    OutFH.write("%s\n" % PML)
    
def SetupPyMOLObjectNames(FileIndex):
    """Setup hierarchy of PyMOL groups and objects for ligand centric views of
    chains and ligands present in input file.
    """

    PyMOLObjectNames = {}
    PyMOLObjectNames["Chains"] = {}
    PyMOLObjectNames["Ligands"] = {}

    # Setup groups and objects for complex...
    SetupPyMOLObjectNamesForComplex(FileIndex, PyMOLObjectNames)
    
    # Setup groups and objects for chain...
    SpecifiedChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]
    for ChainID in SpecifiedChainsAndLigandsInfo["ChainIDs"]:
        SetupPyMOLObjectNamesForChain(FileIndex, PyMOLObjectNames, ChainID)
        
        # Setup groups and objects for ligand...
        for LigandID in SpecifiedChainsAndLigandsInfo["LigandIDs"][ChainID]:
            SetupPyMOLObjectNamesForLigand(FileIndex, PyMOLObjectNames, ChainID, LigandID)

    return PyMOLObjectNames

def SetupPyMOLObjectNamesForComplex(FileIndex, PyMOLObjectNames):
    """Setup groups and objects for complex. """
    
    PDBFileRoot = OptionsInfo["InfilesInfo"]["InfilesRoots"][FileIndex]
    
    PDBGroupName = "%s" % PDBFileRoot
    PyMOLObjectNames["PDBGroup"] = PDBGroupName
    PyMOLObjectNames["PDBGroupMembers"] = []

    ComplexGroupName = "%s.Complex" % PyMOLObjectNames["PDBGroup"]
    PyMOLObjectNames["ComplexGroup"] = ComplexGroupName
    PyMOLObjectNames["PDBGroupMembers"].append(ComplexGroupName)
    
    PyMOLObjectNames["Complex"] = "%s.Complex" % ComplexGroupName
    if OptionsInfo["SurfaceComplex"]:
        PyMOLObjectNames["ComplexHydrophobicSurface"] = "%s.Surface" % ComplexGroupName

    PyMOLObjectNames["ComplexGroupMembers"] = []
    PyMOLObjectNames["ComplexGroupMembers"].append(PyMOLObjectNames["Complex"])
    if OptionsInfo["SurfaceComplex"]:
        PyMOLObjectNames["ComplexGroupMembers"].append(PyMOLObjectNames["ComplexHydrophobicSurface"])
    
def SetupPyMOLObjectNamesForChain(FileIndex, PyMOLObjectNames, ChainID):
    """Setup groups and objects for chain."""
    
    PDBGroupName = PyMOLObjectNames["PDBGroup"]
    
    PyMOLObjectNames["Chains"][ChainID] = {}
    PyMOLObjectNames["Ligands"][ChainID] = {}
    
    # Set up chain group and chain objects...
    ChainGroupName = "%s.Chain%s" % (PDBGroupName, ChainID)
    PyMOLObjectNames["Chains"][ChainID]["ChainGroup"] = ChainGroupName
    PyMOLObjectNames["PDBGroupMembers"].append(ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"] = []
    
    # Setup chain complex group and objects...
    ChainComplexGroupName = "%s.Complex" % (ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroup"] = ChainComplexGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainComplexGroupName)

    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"] = []
    
    Name = "%s.Complex" % (ChainComplexGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainComplex"] = Name
    PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"].append(Name)
    
    if OptionsInfo["SurfaceChainComplex"]:
        Name = "%s.Surface" % (ChainComplexGroupName)
        PyMOLObjectNames["Chains"][ChainID]["ChainComplexHydrophobicSurface"] = Name
        PyMOLObjectNames["Chains"][ChainID]["ChainComplexGroupMembers"].append(Name)

    # Setup up a group for individual chains...
    ChainAloneGroupName = "%s.Chain" % (ChainGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroup"] = ChainAloneGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainAloneGroupName)
        
    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"] = []
        
    Name = "%s.Chain" % (ChainAloneGroupName)
    PyMOLObjectNames["Chains"][ChainID]["ChainAlone"] = Name
    PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"].append(Name)

    if GetChainAloneResidueTypesStatus(FileIndex, ChainID):
        # Setup residue type group and its subgroups...
        ResiduesGroupName = "%s.Residues" % (ChainAloneGroupName)
        
        ResiduesGroupIDPrefix = "ChainAloneResidues"
        ResiduesGroupID = "%sGroup" % ResiduesGroupIDPrefix

        # Add residue group to chain alone group...
        PyMOLObjectNames["Chains"][ChainID][ResiduesGroupID] = ResiduesGroupName
        PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"].append(ResiduesGroupName)
        
        # Initialize residue group members...
        ResiduesGroupMembersID = "%sGroupMembers" % ResiduesGroupIDPrefix
        PyMOLObjectNames["Chains"][ChainID][ResiduesGroupMembersID] = []

        # Setup residues sub groups and its members...
        for SubGroupType in ["Aromatic", "Hydrophobic", "Polar", "Positively_Charged", "Negatively_Charged", "Other"]:
            SubGroupID = re.sub("_", "", SubGroupType)

            ResiduesSubGroupName = "%s.%s" % (ResiduesGroupName, SubGroupType)
            ResiduesSubGroupID = "%s%sGroup" % (ResiduesGroupIDPrefix, SubGroupID)

            # Add sub group to residues group...
            PyMOLObjectNames["Chains"][ChainID][ResiduesSubGroupID] = ResiduesSubGroupName
            PyMOLObjectNames["Chains"][ChainID][ResiduesGroupMembersID].append(ResiduesSubGroupName)

            # Initialize sub group members...
            ResiduesSubGroupMembersID = "%s%sGroupMembers" % (ResiduesGroupIDPrefix, SubGroupID)
            PyMOLObjectNames["Chains"][ChainID][ResiduesSubGroupMembersID] = []
            
            # Add sub group members to subgroup...
            for MemberType in ["Residues", "Surface"]:
                MemberID = re.sub("_", "", MemberType)

                SubGroupMemberName = "%s.%s" % (ResiduesSubGroupName, MemberType)
                SubGroupMemberID = "%s%s%s" % (ResiduesGroupIDPrefix, SubGroupID, MemberID)
                
                PyMOLObjectNames["Chains"][ChainID][SubGroupMemberID] = SubGroupMemberName
                PyMOLObjectNames["Chains"][ChainID][ResiduesSubGroupMembersID].append(SubGroupMemberName)

    if GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
        # Setup cavity and surface groups...
        for SubGroupType in ["Cavities", "Surface"]:
            SubGroupName = "%s.%s" % (ChainAloneGroupName, SubGroupType)
            SubGroupID = "ChainAlone%sGroup" % (SubGroupType)
            SubGroupMembersID = "%sMembers" % (SubGroupID)

            PyMOLObjectNames["Chains"][ChainID][SubGroupID] = SubGroupName
            PyMOLObjectNames["Chains"][ChainID]["ChainAloneGroupMembers"].append(SubGroupName)
            
            PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID] = []
            
            # Setup a generic color surface...
            SurfaceName = "%s.Surface" % (SubGroupName)
            SurfaceID = "%sSurface" % (SubGroupID)
            PyMOLObjectNames["Chains"][ChainID][SurfaceID] = SurfaceName
            PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID].append(SurfaceName)
        
            if GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
                # Setup hydrophobicity surface...
                HydrophobicSurfaceName = "%s.Hydrophobicity" % (SubGroupName)
                HydrophobicSurfaceID = "%sHydrophobicSurface" % (SubGroupID)
                PyMOLObjectNames["Chains"][ChainID][HydrophobicSurfaceID] = HydrophobicSurfaceName
                PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID].append(HydrophobicSurfaceName)
                
                # Setup hydrophobicity and charge surface...
                HydrophobicChargeSurfaceName = "%s.Hydrophobicity_Charge" % (SubGroupName)
                HydrophobicChargeSurfaceID = "%sHydrophobicChargeSurface" % (SubGroupID)
                PyMOLObjectNames["Chains"][ChainID][HydrophobicChargeSurfaceID] = HydrophobicChargeSurfaceName
                PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID].append(HydrophobicChargeSurfaceName)
                
            if GetChainAloneSurfaceChainElectrostaticsStatus(FileIndex, ChainID):
                # Setup electrostatics group...
                ElectrostaticsGroupName = "%s.Vacuum_Electrostatics" % (SubGroupName)
                ElectrostaticsGroupID = "%sElectrostaticsGroup" % (SubGroupID)
                ElectrostaticsGroupMembersID = "%sElectrostaticsGroupMembers" % (SubGroupID)
                
                PyMOLObjectNames["Chains"][ChainID][ElectrostaticsGroupID] = ElectrostaticsGroupName
                PyMOLObjectNames["Chains"][ChainID][SubGroupMembersID].append(ElectrostaticsGroupName)
                
                # Setup electrostatics group members...
                PyMOLObjectNames["Chains"][ChainID][ElectrostaticsGroupMembersID] = []
                        
                for MemberType in ["Chain", "Contact_Potential", "Map", "Legend"]:
                    MemberID = re.sub("_", "", MemberType)
                    
                    Name = "%s.%s" % (ElectrostaticsGroupName, MemberType)
                    NameID = "%s%s" % (ElectrostaticsGroupID, MemberID)
                
                    PyMOLObjectNames["Chains"][ChainID][NameID] = Name
                    PyMOLObjectNames["Chains"][ChainID][ElectrostaticsGroupMembersID].append(Name)
    
    # Setup solvent and inorganic objects for chain...
    for NameID in ["Solvent", "Inorganic"]:
        Name = "%s.%s" % (ChainGroupName, NameID)
        PyMOLObjectNames["Chains"][ChainID][NameID] = Name
        PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(Name)
    
def SetupPyMOLObjectNamesForLigand(FileIndex, PyMOLObjectNames, ChainID, LigandID):
    """Stetup groups and objects for ligand."""

    PyMOLObjectNames["Ligands"][ChainID][LigandID] = {}
    
    ChainGroupName = PyMOLObjectNames["Chains"][ChainID]["ChainGroup"]
    
    # Setup a chain level ligand group...
    ChainLigandGroupName = "%s.Ligand%s" % (ChainGroupName, LigandID)
    PyMOLObjectNames["Ligands"][ChainID][LigandID]["ChainLigandGroup"] = ChainLigandGroupName
    PyMOLObjectNames["Chains"][ChainID]["ChainGroupMembers"].append(ChainLigandGroupName)
    
    PyMOLObjectNames["Ligands"][ChainID][LigandID]["ChainLigandGroupMembers"] = []

    # Set up ligand group and its members...
    GroupName = "%s.Ligand" % (ChainLigandGroupName)
    GroupNameID = "LigandGroup"
    GroupMembersID = "LigandGroupMembers"
    
    PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupNameID] = GroupName
    PyMOLObjectNames["Ligands"][ChainID][LigandID]["ChainLigandGroupMembers"].append(GroupName)
    
    LigandName = "%s.Ligand" % (GroupName)
    LigandNameID = "Ligand"
    PyMOLObjectNames["Ligands"][ChainID][LigandID][LigandNameID] = LigandName
    
    PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupMembersID] = []
    PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupMembersID].append(LigandName)
    
    # Add ball and stick...
    BallAndStickName = "%s.BallAndStick" % (GroupName)
    BallAndStickID = "LigandBallAndStick"
    PyMOLObjectNames["Ligands"][ChainID][LigandID][BallAndStickID] = BallAndStickName
    PyMOLObjectNames["Ligands"][ChainID][LigandID][GroupMembersID].append(BallAndStickName)

def RetrieveInfilesInfo():
    """Retrieve information for input files."""

    InfilesInfo = {}
    
    InfilesInfo["InfilesNames"] = []
    InfilesInfo["InfilesRoots"] = []
    InfilesInfo["ChainsAndLigandsInfo"] = []
    
    for Infile in OptionsInfo["InfilesNames"]:
        FileDir, FileName, FileExt = MiscUtil.ParseFileName(Infile)
        InfileRoot = FileName
        
        ChainsAndLigandInfo = PyMOLUtil.GetChainsAndLigandsInfo(Infile, InfileRoot)
        
        InfilesInfo["InfilesNames"].append(Infile)
        InfilesInfo["InfilesRoots"].append(InfileRoot)
        InfilesInfo["ChainsAndLigandsInfo"].append(ChainsAndLigandInfo)
    
    OptionsInfo["InfilesInfo"] = InfilesInfo

def RetrieveRefFileInfo():
    """Retrieve information for ref file."""

    RefFileInfo = {}
    if not OptionsInfo["Align"]:
        OptionsInfo["RefFileInfo"] = RefFileInfo
        return

    RefFile = OptionsInfo["RefFileName"]
    
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(RefFile)
    RefFileRoot = FileName
    
    if re.match("^FirstInputFile$", OptionsInfo["AlignRefFile"], re.I):
        ChainsAndLigandInfo = OptionsInfo["InfilesInfo"]["ChainsAndLigandsInfo"][0]
    else:
        MiscUtil.PrintInfo("\nRetrieving chain and ligand information for alignment reference file %s..." % RefFile)
        ChainsAndLigandInfo = PyMOLUtil.GetChainsAndLigandsInfo(RefFile, RefFileRoot)

    RefFileInfo["RefFileName"] = RefFile
    RefFileInfo["RefFileRoot"] = RefFileRoot
    RefFileInfo["PyMOLObjectName"] = "AlignRef_%s" % RefFileRoot
    RefFileInfo["ChainsAndLigandsInfo"] = ChainsAndLigandInfo
    
    OptionsInfo["RefFileInfo"] = RefFileInfo

def ProcessChainAndLigandIDs():
    """Process specified chain and ligand IDs for infiles."""
    
    OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"] = []
    
    for FileIndex in range(0, len(OptionsInfo["InfilesInfo"]["InfilesNames"])):
        MiscUtil.PrintInfo("\nProcessing specified chain and ligand IDs for input file %s..." % OptionsInfo["InfilesInfo"]["InfilesNames"][FileIndex])
        
        ChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["ChainsAndLigandsInfo"][FileIndex]
        SpecifiedChainsAndLigandsInfo = PyMOLUtil.ProcessChainsAndLigandsOptionsInfo(ChainsAndLigandsInfo, "-c, --chainIDs", OptionsInfo["ChainIDs"], "-l, --ligandIDs", OptionsInfo["LigandIDs"])
        ProcessResidueTypesAndSurfaceOptions(FileIndex, SpecifiedChainsAndLigandsInfo)
        OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"].append(SpecifiedChainsAndLigandsInfo)
        
        CheckPresenceOfValidLigandIDs(ChainsAndLigandsInfo, SpecifiedChainsAndLigandsInfo)
        
def ProcessResidueTypesAndSurfaceOptions(FileIndex, SpecifiedChainsAndLigandsInfo):
    """Process residue types and surface options for chains."""

    SpecifiedChainsAndLigandsInfo["ChainSurfaces"] = {}
    SpecifiedChainsAndLigandsInfo["SurfaceChain"] = {}
    SpecifiedChainsAndLigandsInfo["SurfaceChainElectrostatics"] = {}
    
    SpecifiedChainsAndLigandsInfo["ResidueTypesChain"] = {}

    # Load infile...
    Infile = OptionsInfo["InfilesInfo"]["InfilesNames"][FileIndex]
    MolName = OptionsInfo["InfilesInfo"]["InfilesRoots"][FileIndex]
    pymol.cmd.load(Infile, MolName)
    
    for ChainID in SpecifiedChainsAndLigandsInfo["ChainIDs"]:
        AminoAcidsPresent = PyMOLUtil.AreAminoAcidResiduesPresent(MolName, ChainID)

        # Process surfaces for chains...
        if re.match("^auto$", OptionsInfo["SurfaceChain"], re.I):
            SurfaceChain = True if AminoAcidsPresent else False
        else:
            SurfaceChain = True if re.match("^yes$", OptionsInfo["SurfaceChain"], re.I) else False
        SpecifiedChainsAndLigandsInfo["SurfaceChain"][ChainID] = SurfaceChain

        if re.match("^auto$", OptionsInfo["SurfaceChainElectrostatics"], re.I):
            SurfaceChainElectrostatics = True if AminoAcidsPresent else False
        else:
            SurfaceChainElectrostatics = True if re.match("^yes$", OptionsInfo["SurfaceChainElectrostatics"], re.I) else False
        SpecifiedChainsAndLigandsInfo["SurfaceChainElectrostatics"][ChainID] = SurfaceChainElectrostatics

        # A generic color surafce is always created...
        ChainSurfaces = True
        SpecifiedChainsAndLigandsInfo["ChainSurfaces"][ChainID] = ChainSurfaces
        
        # Process residue types for chains...
        if re.match("^auto$", OptionsInfo["ResidueTypesChain"], re.I):
            ResidueTypesChain = True if AminoAcidsPresent else False
        else:
            ResidueTypesChain = True if re.match("^yes$", OptionsInfo["ResidueTypesChain"], re.I) else False
        SpecifiedChainsAndLigandsInfo["ResidueTypesChain"][ChainID] = ResidueTypesChain

    # Delete loaded object...
    pymol.cmd.delete(MolName)

def GetChainAloneResidueTypesStatus(FileIndex, ChainID):
    """Get status of residue types for chain alone object."""

    Status = True if OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]["ResidueTypesChain"][ChainID] else False
    
    return Status

def GetChainAloneContainsSurfacesStatus(FileIndex, ChainID):
    """Get status of surfaces present in chain alone object."""

    return OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]["ChainSurfaces"][ChainID]

def GetChainAloneSurfaceChainStatus(FileIndex, ChainID):
    """Get status of hydrophobic surfaces for chain alone object."""

    return OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]["SurfaceChain"][ChainID]

def GetChainAloneSurfaceChainElectrostaticsStatus(FileIndex, ChainID):
    """Get status of electrostatics surfaces for chain alone object."""

    return OptionsInfo["InfilesInfo"]["SpecifiedChainsAndLigandsInfo"][FileIndex]["SurfaceChainElectrostatics"][ChainID]

def CheckPresenceOfValidLigandIDs(ChainsAndLigandsInfo, SpecifiedChainsAndLigandsInfo):
    """Check presence of valid ligand IDs."""

    MiscUtil.PrintInfo("\nSpecified chain IDs: %s" % (", ".join(SpecifiedChainsAndLigandsInfo["ChainIDs"])))
    
    for ChainID in SpecifiedChainsAndLigandsInfo["ChainIDs"]:
        if len (SpecifiedChainsAndLigandsInfo["LigandIDs"][ChainID]):
            MiscUtil.PrintInfo("Chain ID: %s; Specified LigandIDs: %s" % (ChainID, ", ".join(SpecifiedChainsAndLigandsInfo["LigandIDs"][ChainID])))
        else:
            MiscUtil.PrintInfo("Chain IDs: %s; Specified LigandIDs: None" % (ChainID))
            MiscUtil.PrintWarning("No valid ligand IDs found for chain ID, %s. PyMOL groups and objects related to ligand and binding pockect won't be created." % (ChainID))

def RetrieveFirstChainID(FileIndex):
    """Get first chain ID."""
    
    ChainsAndLigandsInfo = OptionsInfo["InfilesInfo"]["ChainsAndLigandsInfo"][FileIndex]
    
    FirstChainID = None
    if len(ChainsAndLigandsInfo["ChainIDs"]):
        FirstChainID = ChainsAndLigandsInfo["ChainIDs"][0]
    
    return FirstChainID

def ProcessResidueTypes():
    """Process residue types. """

    ResidueTypesNamesInfo, ResidueTypesParamsInfo = PyMOLUtil.ProcessResidueTypesOptionsInfo("-r, --residueTypes", OptionsInfo["ResidueTypes"])
    OptionsInfo["ResidueTypesNames"] = ResidueTypesNamesInfo
    OptionsInfo["ResidueTypesParams"] = ResidueTypesParamsInfo
    
def ProcessSurfaceAtomTypesColors():
    """Process surface atom types colors. """

    AtomTypesColorNamesInfo = PyMOLUtil.ProcessSurfaceAtomTypesColorsOptionsInfo("--surfaceAtomTypesColors", OptionsInfo["SurfaceAtomTypesColors"])
    OptionsInfo["AtomTypesColorNames"] = AtomTypesColorNamesInfo

def ProcessOptions():
    """Process and validate command line arguments and options"""

    MiscUtil.PrintInfo("Processing options...")
    
    # Validate options...
    ValidateOptions()
    
    OptionsInfo["Align"] = True if re.match("^Yes$", Options["--align"], re.I) else False
    OptionsInfo["AlignMethod"] = Options["--alignMethod"].lower()
    OptionsInfo["AlignMode"] = Options["--alignMode"]
    
    OptionsInfo["AllowEmptyObjects"] = True if re.match("^Yes$", Options["--allowEmptyObjects"], re.I) else False

    OptionsInfo["CavityCulled"] = True if re.match("^Yes$", Options["--cavityCulled"], re.I) else False
    OptionsInfo["SurfaceCavityMode"] = 2 if OptionsInfo["CavityCulled"] else 1
    
    OptionsInfo["CavityCullSize"] = float(Options["--cavityCullSize"])
    OptionsInfo["CavityCutoff"] = float(Options["--cavityCutoff"])
    OptionsInfo["CavityRadius"] = float(Options["--cavityRadius"])
    
    OptionsInfo["Infiles"] = Options["--infiles"]
    OptionsInfo["InfilesNames"] =  Options["--infileNames"]

    OptionsInfo["AlignRefFile"] = Options["--alignRefFile"]
    if re.match("^FirstInputFile$", Options["--alignRefFile"], re.I):
        OptionsInfo["RefFileName"] = OptionsInfo["InfilesNames"][0]
    else:
        OptionsInfo["RefFileName"] = Options["--alignRefFile"]
    
    OptionsInfo["Overwrite"] = Options["--overwrite"]
    OptionsInfo["PMLOut"] = True if re.match("^Yes$", Options["--PMLOut"], re.I) else False
    
    OptionsInfo["Outfile"] = Options["--outfile"]
    FileDir, FileName, FileExt = MiscUtil.ParseFileName(OptionsInfo["Outfile"])
    OptionsInfo["PSEOut"] = False 
    if re.match("^pml$", FileExt, re.I):
        OptionsInfo["PMLOutfile"] = OptionsInfo["Outfile"] 
        OptionsInfo["PMEOutfile"] = re.sub(".pml$", ".pme", OptionsInfo["Outfile"]) 
    elif re.match("^pse$", FileExt, re.I):
        OptionsInfo["PSEOut"] = True 
        OptionsInfo["PSEOutfile"] = OptionsInfo["Outfile"] 
        OptionsInfo["PMLOutfile"] = re.sub(".pse$", ".pml", OptionsInfo["Outfile"]) 
        if os.path.exists(OptionsInfo["PMLOutfile"]) and (not OptionsInfo["Overwrite"]):
            MiscUtil.PrintError("The intermediate output file to be generated, %s, already exist. Use option \"--ov\" or \"--overwrite\" and try again." % OptionsInfo["PMLOutfile"] )

    OptionsInfo["LabelFontID"] = int(Options["--labelFontID"])
    
    OptionsInfo["ResidueTypesChain"] = Options["--residueTypesChain"]
    OptionsInfo["ResidueTypes"] = Options["--residueTypes"]
    ProcessResidueTypes()
    
    OptionsInfo["SurfaceChain"] = Options["--surfaceChain"]
    OptionsInfo["SurfaceChainElectrostatics"] = Options["--surfaceChainElectrostatics"]
    
    OptionsInfo["SurfaceChainComplex"] = True if re.match("^Yes$", Options["--surfaceChainComplex"], re.I) else False
    OptionsInfo["SurfaceComplex"] = True if re.match("^Yes$", Options["--surfaceComplex"], re.I) else False
    
    # Retrieve surface colors for generic surfaces..
    SurfaceColors = re.sub(" ", "", Options["--surfaceColors"])
    SurfaceColorsWords = SurfaceColors.split(",")
    if len(SurfaceColorsWords) != 2:
        MiscUtil.PrintError("The number of comma delinited color names, %d, specified using \"--surfaceColors\" option, \"%s\",  must be a 2." % (len(SurfaceColorsWords), Options["--surfaceColors"]))
    OptionsInfo["SurfaceColors"] = SurfaceColors
    OptionsInfo["SurfaceCavityColor"] = SurfaceColorsWords[0]
    OptionsInfo["SurfaceColor"] = SurfaceColorsWords[1]
    
    OptionsInfo["SurfaceColorPalette"] = Options["--surfaceColorPalette"]
    OptionsInfo["SurfaceAtomTypesColors"] = Options["--surfaceAtomTypesColors"]
    ProcessSurfaceAtomTypesColors()
    
    OptionsInfo["SurfaceTransparency"] = float(Options["--surfaceTransparency"])
    
    RetrieveInfilesInfo()
    RetrieveRefFileInfo()
    
    OptionsInfo["ChainIDs"] = Options["--chainIDs"]
    OptionsInfo["LigandIDs"] = Options["--ligandIDs"]
    
    ProcessChainAndLigandIDs()

def RetrieveOptions(): 
    """Retrieve command line arguments and options"""
    
    # Get options...
    global Options
    Options = docopt(_docoptUsage_)

    # Set current working directory to the specified directory...
    WorkingDir = Options["--workingdir"]
    if WorkingDir:
        os.chdir(WorkingDir)
    
    # Handle examples option...
    if "--examples" in Options and Options["--examples"]:
        MiscUtil.PrintInfo(MiscUtil.GetExamplesTextFromDocOptText(_docoptUsage_))
        sys.exit(0)

def ValidateOptions():
    """Validate option values"""
    
    MiscUtil.ValidateOptionTextValue("--align", Options["--align"], "yes no")
    MiscUtil.ValidateOptionTextValue("--alignMethod", Options["--alignMethod"], "align cealign super")
    MiscUtil.ValidateOptionTextValue("--alignMode", Options["--alignMode"], "FirstChain Complex")
    
    MiscUtil.ValidateOptionTextValue("--allowEmptyObjects", Options["--allowEmptyObjects"], "yes no")

    MiscUtil.ValidateOptionTextValue("--cavityCulled", Options["--cavityCulled"], "yes no")
    MiscUtil.ValidateOptionFloatValue("--cavityCullSize", Options["--cavityCullSize"], {">": 0.0})
    MiscUtil.ValidateOptionFloatValue("--cavityCutoff", Options["--cavityCutoff"], {">": 0.0})
    MiscUtil.ValidateOptionFloatValue("--cavityRadius", Options["--cavityRadius"], {">": 0.0})
    
    # Expand infiles to handle presence of multiple input files...
    InfileNames = MiscUtil.ExpandFileNames(Options["--infiles"], ",")
    if not len(InfileNames):
        MiscUtil.PrintError("No input files specified for \"-i, --infiles\" option")

    # Validate file extensions...
    for Infile in InfileNames:
        MiscUtil.ValidateOptionFilePath("-i, --infiles", Infile)
        MiscUtil.ValidateOptionFileExt("-i, --infiles", Infile, "pdb cif")
        MiscUtil.ValidateOptionsDistinctFileNames("-i, --infiles", Infile, "-o, --outfile", Options["--outfile"])
    Options["--infileNames"] = InfileNames
    
    MiscUtil.ValidateOptionFileExt("-o, --outfile", Options["--outfile"], "pml pse")
    MiscUtil.ValidateOptionsOutputFileOverwrite("-o, --outfile", Options["--outfile"], "--overwrite", Options["--overwrite"])

    if re.match("^yes$", Options["--align"], re.I):
        if not re.match("^FirstInputFile$", Options["--alignRefFile"], re.I):
            AlignRefFile = Options["--alignRefFile"]
            MiscUtil.ValidateOptionFilePath("--alignRefFile", AlignRefFile)
            MiscUtil.ValidateOptionFileExt("--alignRefFile", AlignRefFile, "pdb cif")
            MiscUtil.ValidateOptionsDistinctFileNames("--AlignRefFile", AlignRefFile, "-o, --outfile", Options["--outfile"])
    
    MiscUtil.ValidateOptionTextValue("--PMLOut", Options["--PMLOut"], "yes no")
    MiscUtil.ValidateOptionIntegerValue("--labelFontID", Options["--labelFontID"], {})

    MiscUtil.ValidateOptionTextValue("--residueTypesChain", Options["--residueTypesChain"], "yes no auto")
    
    MiscUtil.ValidateOptionTextValue("--surfaceChain", Options["--surfaceChain"], "yes no auto")
    MiscUtil.ValidateOptionTextValue("--surfaceComplex", Options["--surfaceComplex"], "yes no")
    MiscUtil.ValidateOptionTextValue("--surfaceChainComplex", Options["--surfaceChainComplex"], "yes no")
    MiscUtil.ValidateOptionTextValue("--surfaceChainElectrostatics", Options["--surfaceChainElectrostatics"], "yes no auto")
    
    MiscUtil.ValidateOptionTextValue("--surfaceColorPalette", Options["--surfaceColorPalette"], "RedToWhite WhiteToGreen")
    MiscUtil.ValidateOptionFloatValue("--surfaceTransparency", Options["--surfaceTransparency"], {">=": 0.0, "<=": 1.0})
    
# Setup a usage string for docopt...
_docoptUsage_ = """
PyMOLVisualizeCavities.py - Visualize cavities and pockets in macromolecules

Usage:
    PyMOLVisualizeCavities.py [--align <yes or no>] [--alignMethod <align, cealign, super>]
                                    [--alignMode <FirstChain or Complex>] [--alignRefFile <filename>]
                                    [--allowEmptyObjects <yes or no>] [--cavityCulled <Yes or No>]
                                    [--cavityCullSize <number>] [--cavityCutoff <number>] [--cavityRadius <number>]
                                    [--chainIDs <First, All or ID1,ID2...>] [--labelFontID <number>]
                                    [--ligandIDs <Largest, All or ID1,ID2...> ] [--PMLOut <yes or no>]
                                    [--residueTypes <Type,Color,ResNames,...>] [--residueTypesChain <yes or no>]
                                    [--surfaceChain <yes or no>] [--surfaceChainElectrostatics <yes or no>]
                                    [--surfaceChainComplex <yes or no>] [--surfaceComplex <yes or no>]
                                    [--surfaceAtomTypesColors <ColorType,ColorSpec,...>]
                                    [--surfaceColors <ColorName1,ColorName2>] [--surfaceColorPalette <RedToWhite or WhiteToGreen>]
                                    [--surfaceTransparency <number>] [--overwrite] [-w <dir>] -i <infile1,infile2,infile3...> -o <outfile>
    PyMOLVisualizeCavities.py -h | --help | -e | --examples

Description:
    Generate PyMOL visualization files for viewing cavities and pockets in
    macromolecules including proteins and nucleic acids.

    The supported input file format are: PDB (.pdb), CIF (.cif)

    The supported output file formats are: PyMOL script file (.pml), PyMOL session
    file (.pse)

    A variety of PyMOL groups and objects may be  created for visualization of
    cavities and pockets in macromolecules. These groups and objects correspond
    to complexes, surfaces, chains, ligands, inorganics, cavities, and pockets.
    A complete hierarchy of all possible PyMOL groups and objects is shown below:
    
        <PDBFileRoot>
            .Complex
                .Complex
                .Surface
            .Chain<ID>
                .Complex
                    .Complex
                    .Surface
                .Chain
                    .Chain
                    .Residues
                        .Aromatic
                            .Residues
                            .Surface
                        .Hydrophobic
                            .Residues
                            .Surface
                        .Polar
                            .Residues
                            .Surface
                        .Positively_Charged
                            .Residues
                            .Surface
                        .Negatively_Charged
                            .Residues
                            .Surface
                        .Other
                            .Residues
                            .Surface
                    .Cavities
                        .Surface
                        .Hydrophobicity
                        .Hydrophobicity_Charge
                        .Vacuum_Electrostatics
                            .Contact_Potentials
                            .Map
                            .Legend
                    .Surface
                        .Surface
                        .Hydrophobicity
                        .Hydrophobicity_Charge
                        .Vacuum_Electrostatics
                            .Contact_Potentials
                            .Map
                            .Legend
                .Solvent
                .Inorganic
                .Ligand<ID>
                    .Ligand
                        .Ligand
                        .BallAndStick
                .Ligand<ID>
                    .Ligand
                        ... ... ...
            .Chain<ID>
                ... ... ...
                .Ligand<ID>
                    ... ... ...
                .Ligand<ID>
                    ... ... ...
            .Chain<ID>
                ... ... ...
        <PDBFileRoot>
            .Complex
                ... ... ...
            .Chain<ID>
                ... ... ...
                .Ligand<ID>
                    ... ... ...
                .Ligand<ID>
                    ... ... ...
            .Chain<ID>
                ... ... ...
    
    The hydrophobic and electrostatic surfaces are not created for complete complex
    and chain complex in input file(s) by default. A word to the wise: The creation of
    surface objects may slow down loading of PML file and generation of PSE file, based
    on the size of input complexes. The generation of PSE file may also fail.

Options:
    -a, --align <yes or no>  [default: no]
        Align input files to a reference file before visualization.
    --alignMethod <align, cealign, super>  [default: super]
        Alignment methodology to use for aligning input files to a
        reference file.
    --alignMode <FirstChain or Complex>  [default: FirstChain]
        Portion of input and reference files to use for spatial alignment of
        input files against reference file.  Possible values: FirstChain or
        Complex.
        
        The FirstChain mode allows alignment of the first chain in each input
        file to the first chain in the reference file along with moving the rest
        of the complex to coordinate space of the reference file. The complete
        complex in each input file is aligned to the complete complex in reference
        file for the Complex mode.
    --alignRefFile <filename>  [default: FirstInputFile]
        Reference input file name. The default is to use the first input file
        name specified using '-i, --infiles' option.
    --allowEmptyObjects <yes or no>  [default: no]
        Allow creation of empty PyMOL objects corresponding to solvent and
        inorganic atom selections across chains and ligands in input file(s). By
        default, the empty objects are marked for deletion.
    --cavityCulled <Yes or No>  [default: Yes]
        Cull cavities and pockets. The cavities and pockets are culled by default.
        This value is used to set of PyMOL parameter surface_cavity_mode as
        shown below:
        
            Yes: 1 (Cavities and Pockets Only)
            No: 2 (Cavities and Pockets Culled)
        
    --cavityCullSize <number>  [default: 2.0]
        Approximate dimension of the cavity in Angstroms for detecting cavities and
        pockets in the interior of a macromolecule. The higher value makes PyMOL less
        sensitive to detection of smaller cavities.
    --cavityCutoff <number>  [default: 3.0]
        Cavity cutoff in terms of number of solvent radii for detecting cavities
        and pockets. This value is used to set value of PyMOL parameter
        surface_cavity_cutoff.
    --cavityRadius <number>  [default: 5.0]
        Cavity detection radius in terms of number of solvent radii for detecting
        cavities and pockets. The detection of larger pockets is ignored for lower
        value for the cavity radius. This value is used to set of PyMOL parameter
        surface_cavity_radius.
    -c, --chainIDs <First, All or ID1,ID2...>  [default: First]
        List of chain IDs to use for visualizing macromolecules. Possible values:
        First, All, or a comma delimited list of chain IDs. The default is to use the
        chain ID for the first chain in each input file.
    -e, --examples
        Print examples.
    -h, --help
        Print this help message.
    -i, --infiles <infile1,infile2,infile3...>
        Input file names.
    --labelFontID <number>  [default: 7]
        Font ID for drawing labels. Default: 7 (Sans Bold). Valid values: 5 to 16.
        The specified value must be a valid PyMOL font ID. No validation is
        performed. The complete lists of valid font IDs is available at:
        pymolwiki.org/index.php/Label_font_id. Examples: 5 - Sans;
        7 - Sans Bold; 9 - Serif; 10 - Serif Bold.
    -l, --ligandIDs <Largest, All or ID1,ID2...>  [default: All]
        List of ligand IDs to show in chains during visualizing of cavities in
        macromolecules. Possible values: Largest, All, or a comma delimited
        list of ligand IDs. The default is to show all ligands present in all or
        specified chains in each input file.
        
        Ligands are identified using organic selection operator available in PyMOL.
        It'll also  identify buffer molecules as ligands. The largest ligand contains
        the highest number of heavy atoms.
    -o, --outfile <outfile>
        Output file name.
    -p, --PMLOut <yes or no>  [default: yes]
        Save PML file during generation of PSE file.
    -r, --residueTypes <Type,Color,ResNames,...>  [default: auto]
        Residue types, colors, and names to generate for residue groups during
        '--residueTypesChain' option. It is only valid for amino acids.
        
        It is a triplet of comma delimited list of amino acid residues type, residues
        color, and a space delimited list three letter residue names. 
        
        The default values for residue type, color, and name triplets  are shown
        below:
            
            Aromatic,brightorange,HIS PHE TRP TYR,
            Hydrophobic,orange,ALA GLY VAL LEU ILE PRO MET,
            Polar,palegreen,ASN GLN SER THR CYS,
            Positively_Charged,marine,ARG LYS,
            Negatively_Charged,red,ASP GLU
            
        The color name must be a valid PyMOL name. No validation is performed.
        An amino acid name may appear across multiple residue types. All other
        residues are grouped under 'Other'.
    --residueTypesChain <yes or no>  [default: auto]
        Chain residue types. The residue groups are generated using residue types,
        colors, and names specified by '--residueTypes' option. It is only valid for
        amino acids.  By default, the residue type groups are automatically created
        for chains containing amino acids and skipped for chains only containing
        nucleic acids.
    --surfaceChain <yes or no>  [default: auto]
        Surfaces around individual chain colored by hydrophobicity alone and
        both hydrophobicity and charge. The hydrophobicity surface is colored
        at residue level using Eisenberg hydrophobicity scale for residues and color
        gradient specified by '--surfaceColorPalette' option. The  hydrophobicity and
        charge surface is colored [ REF 140 ] at atom level using colors specified for
        groups of atoms by '--surfaceAtomTypesColors' option. This scheme allows
        simultaneous mapping of hyrophobicity and charge values on the surfaces.
        
        This option is only valid for amino acids. By default, both surfaces are
        automatically created for chains containing amino acids and skipped for
        chains containing only nucleic acids.
        
        In addition, generic surfaces colored by '--surfaceColor' are always created
        for chain residues containing amino acids and nucleic acids.
    --surfaceChainElectrostatics <yes or no>  [default: auto]
        Vacuum electrostatics contact potential surface around individual
        chain. A word to the wise from PyMOL documentation: The computed protein
        contact potentials are only qualitatively useful, due to short cutoffs,
        truncation, and lack of solvent "screening".
        
        This option is only valid for amino acids. By default, the electrostatics surface
        is automatically created for chains containing amino acids and
        skipped for chains containing only nucleic acids.
    --surfaceChainComplex <yes or no>  [default: no]
        Hydrophobic surface around chain complex. The  surface is colored by
        hydrophobicity. It is only valid for amino acids.
    --surfaceComplex <yes or no>  [default: no]
        Hydrophobic surface around complete complex. The  surface is colored by
        hydrophobicity. It is only valid for amino acids.
    --surfaceAtomTypesColors <ColorType,ColorSpec,...>  [default: auto]
        Atom colors for generating surfaces colored by hyrophobicity and charge
        around chains and pockets in proteins. It's a pairwise comma delimited list
        of atom color type and color specification for goups of atoms.
        
        The default values for color types [ REF 140 ] along wth color specifications
        are shown below: 
            
            HydrophobicAtomsColor, yellow,
            NegativelyChargedAtomsColor, red,
            PositivelyChargedAtomsColor, blue,
            OtherAtomsColor, gray90
            
        The color names must be valid PyMOL names.
        
        The color values may also be specified as space delimited RGB triplets:
             
            HydrophobicAtomsColor, 0.95 0.78 0.0,
            NegativelyChargedAtomsColor, 1.0 0.4 0.4,
            PositivelyChargedAtomsColor, 0.2 0.5 0.8,
            OtherAtomsColor, 0.95 0.95 0.95
            
    --surfaceColors <ColorName1,ColorName2>  [default: salmon,lightblue]
        Color names for surfaces around cavities and chains. These colors are not
        used for surfaces and cavities colored by hydrophobicity and charge. The
        color names must be valid PyMOL names.
    --surfaceColorPalette <RedToWhite or WhiteToGreen>  [default: RedToWhite]
        Color palette for hydrophobic surfaces around chains and pockets in proteins.
        Possible values: RedToWhite or WhiteToGreen from most hydrophobic amino
        acid to least hydrophobic. The colors values for amino acids are taken from
        color_h script available as part of the Script Library at PyMOL Wiki.
    --surfaceTransparency <number>  [default: 0.25]
        Surface transparency for molecular surfaces.
    --overwrite
        Overwrite existing files.
    -w, --workingdir <dir>
        Location of working directory which defaults to the current directory.

Examples:
    To visualize cavities in the first chain along with the largest ligand in the
    first chain, solvents, and inorganics, in a PDB file, and generate a PML
    file, type:

        % PyMOLVisualizeCavities.py -i Sample4.pdb -o Sample4.pml

    To visualize cavities in all chain along with all ligands, solvents,
    and inorganics, in a PDB file, and generate a PML file, type:

        % PyMOLVisualizeCavities.py -c All -l All -i Sample4.pdb
          -o Sample4.pml

    To visualize cavities in the first chain at a specific cavity radius and cutoff
    using specifc colors for surfaces corresponding to cavities and non-cavities,
    and generate a PML file, type:

        % PyMOLVisualizeCavities.py  --cavityRadius 3 --cavityCutoff 5
           --surfaceColors "red,blue" -i Sample4.pdb -o Sample4.pml

    To visualize cavities in the first chain along with the largest ligand in the
    first chain, solvents, and inorganics, in PDB files, along with aligning first
    chain in each input file to the first chain inand generate a PML file, type:

        % PyMOLVisualizeCavities.py --align yes
          -i "Sample5.pdb,Sample6.pdb,Sample7.pdb"
          -o SampleOut.pml

Author:
    Manish Sud(msud@san.rr.com)

See also:
    DownloadPDBFiles.pl, PyMOLVisualizeCryoEMDensity.py,
    PyMOLVisualizeElectronDensity.py, PyMOLVisualizeInterfaces.py
    PyMOLVisualizeMacromolecules.py, PyMOLVisualizeSurfaceAndBuriedResidues.py

Copyright:
    Copyright (C) 2020 Manish Sud. All rights reserved.

    The functionality available in this script is implemented using PyMOL, a
    molecular visualization system on an open source foundation originally
    developed by Warren DeLano.

    This file is part of MayaChemTools.

    MayaChemTools is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option) any
    later version.

"""

if __name__ == "__main__":
    main()
