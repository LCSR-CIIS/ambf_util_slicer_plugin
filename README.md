# AMBF_Utils Module for 3D Slicer
This module is a collection of utilities for exporting volumes and markers from 3D Slicer to a format that can be read by the AMBF Simulator and various volumetric plugins for the AMBF Simulator.

Primarily, this module is expected to be used with continuum-manipulator-volumetric-drilling-plugin (https://github.com/htp2/continuum-manip-volumetric-drilling-plugin.git) and volumetric-drilling-plugin (https://github.com/LCSR-SICKKIDS/volumetric_drilling)

I would be remiss to not mention the work of Adnan Munawar et al., whose scripts in the referenced volumetric-drilling-plugin were used as a starting point for this module.

## Exporting Volumes (LabelMaps)
Volumes will be exported into a folder of png files sliced in the superior/inferior plane. The orientation, etc. of the volume will be made to match the loading expectations of the AMBF Simulator. Primarily this means that the volume will be in LPS coordinates, rotated by pi/2 in the z axis, and will have its origin represented at the center of the volume, rather than the corner. NOTE: someday, AMBF may change their loading conventions, and this module will need to be updated to match.

You may export a volume as a "grayscale" image, or it will be exported using the current color map of the segmentation.

The "volume" must be a labelmap (which you can create by segmenting a volume in 3D Slicer). 

You can generate the accompanying AMBF ADF yaml file by checking the "Generate AMBF yaml" checkbox. This will generate a yaml file with the same name as the volume in the output directory which will have the size information, etc. setup so that the AMBF Simulator can read the volume.


## Exporting Markups
Markups (for now markup fiducials and markup curves are supported) will be to a csv file. A reference volume must be selected, and the markups will be exported in the coordinate system of the reference volume once loaded into AMBF. 

## Example Usage of features
- Pick a volume of your choice, and make some segmentation on it (e.g. segment out bone from a CT volume)
- Convert the segmentation to a labelmap, and use this module to export the labelmap to a folder of png files, with a yaml file for the AMBF Simulator
- Draw a curve on the volume, and export it to a csv file
- Load the volume into the AMBF simulator (e.g. using the volumetric-drilling-plugin). Use the ambf_trace_plugin (https://github.com/htp2/ambf_trace_plugin) to load the curve into the AMBF simulator using ```--csv_filename_static_traces <path_to_csv_file>``` and ```--static_trace_rel_body_name <volume_name>```


## Setup Instructions
After cloning this module:
- Open Slicer
- In the 'Welcome to Slicer' Module, click 'Customize Slicer'
- Press the Modules tab
- Under 'Paths' select 'Add'
- Select folder that contains this README document
- Restart Slicer as prompted
- The module should appear under the group name ```AMBF```
