#!/usr/bin/env python3
"""One Static-Optimization chunk over [t0,t1] of pop60.mot, using the prebuilt reserves model.
Unique tag -> unique output (<tag>_StaticOptimization_activation.sto). For parallel SO.
Args: t0 t1 tag"""
import sys, os
import opensim as osim
HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.environ.get("SO_MODEL") or os.path.join(HERE, "cyclist_min_reserves.osim")
MOT = os.environ.get("CHUNK_MOT") or os.path.join(HERE, "pop60.mot")
t0, t1, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]

so = osim.StaticOptimization()
so.setStartTime(t0); so.setEndTime(t1)
so.setUseModelForceSet(True); so.setActivationExponent(2.0); so.setUseMusclePhysiology(True)

tool = osim.AnalyzeTool()
tool.setName(tag)
tool.setModelFilename(MODEL)
tool.setInitialTime(t0); tool.setFinalTime(t1)
tool.setCoordinatesFileName(MOT)
tool.setLowpassCutoffFrequency(6.0)
tool.setResultsDir(HERE)
tool.getAnalysisSet().cloneAndAppend(so)
setup = os.path.join(HERE, f"{tag}_setup.xml")
tool.printToXML(setup)
osim.AnalyzeTool(setup).run()
print(f"chunk {tag} [{t0},{t1}] done", flush=True)
