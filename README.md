# AMBF_Utils Module for 3D Slicer
This module is a collection of utilities for exporting volumes and markers from 3D Slicer to a format that can be read by the AMBF Simulator and various volumetric plugins for the AMBF Simulator.

Primarily, this module is expected to be used with continuum-manipulator-volumetric-drilling-plugin (https://github.com/htp2/continuum-manip-volumetric-drilling-plugin.git) and volumetric-drilling-plugin (https://github.com/LCSR-SICKKIDS/volumetric_drilling)

I would be remiss to not mention the work of Adnan Munawar et al., whose scripts in the referenced volumetric-drilling-plugin were used as a starting point for this module.

Tested with Slicer 5.2.1. Binary install is fine, no need for compilation from source.

Video Demonstrating Use:

https://user-images.githubusercontent.com/17507145/226940289-da422ef1-dd00-476c-a6f7-c10599b9dded.mp4


## Exporting LabelMaps (Slicer Segmentations) To AMBF Volumes:
Volumes will be exported into a folder of png files sliced in the superior/inferior plane. Their origin (when loaded into AMBF) will be at the center of the volume, and the orientation will be LPS corresponding to x,y,z in its local coordinate system in AMBF. 

To assist with understanding coordinate systems in 3DSlicer, see the following link: https://www.slicer.org/wiki/Coordinate_systems

You may export a volume as a "grayscale" image, or it will be exported using the current color map of the segmentation.

Using Slicer's terminology, an AMBF "volume" really is generally a segmentation made from a volume. For the purposes of this plugin, you must must use a labelmap (which you can create by segmenting a volume in 3D Slicer). From that segmentation, you can right click on the node and "convert segmentation to labelmap"

### Automatically Generate ADF files for AMBF Simulator
You can generate the accompanying AMBF ADF yaml file by checking the "Generate AMBF yaml" checkbox. This will generate a yaml file with the same name as the volume in the output directory which will have the size information, etc. setup so that the AMBF Simulator can read the volume.

In addition to the volume, this configuration file will also contain a body with [volume_name]_anatomical_origin which will be located at the anatomical or "space" origin of the volume. This is the "world" origin according to the imager the image was taken with (e.g. CT scanner, MRI machine, etc.). Usually it is located somewhere outside the image itself. For example, in 3D Slicer your markup coordinates are defined relative to this anatomical origin. The only difference is that the anatomical origin will be defined using the LPS = (x,y,z) convention in AMBF, whereas it is RAS = (x,y,z) in 3D slicer.

You can see this file within the Slicer plugin using the "YAML Output" Tab. If you press "Load/Refresh YAML File", it will update that text from the file into the text box in the plugin. Then, if you want you can make any hard-coded edits and save that out to the file using the "Overwrite YAML file". Note: By design, if you "Export LabelMap to PNGs for AMBF", the yaml file will be overridden to the default values given your data, so do any overwriting last if need be. This editing can also be done in your favorite text editor, this just allows you to do it in the same window.

If you want to regenerate the yaml file (e.g. if you changed a setting like the volume name or scale), and do not want to regenerate all of the image slices again, you may disable this with the "Generate Image Slices" checkbox.

### Applying a non-identity offset to the volume
When you select a LabelMap, a transform called "AMBF_Pose" will be generated in the "Transforms" module. This will correspond to the initial pose that your volume will have in AMBF (i.e. what goes in the ADF yaml file). This is how you can specify your volume to have a non-identity pose. Both the volume's position and the anatomical origin's position will be updated in the ADF file accordingly. Several other Transforms will appear, all having internal relations within 3DSlicer. This is done to allow the "AMBF_Pose" transform to act like an LPS transform at the volume's center (i.e. as if it is a transform applied to the volume in AMBF).

### A Note on the [volume_name]_anatomical_origin body
By default, this body is static in AMBF, and will not move if you were to move the volume and vice versa [NOTE: TODO: There are upcoming changes to AMBF which should allow for a parent-child relationship between these two that will uncomplicate this]. If you plan on moving the volume at any time in your application, you need to be sure to also move this body accordingly, otherwise you will no longer be able to use the anatomical origin as a reference point.

## Exporting Markups
Markups (for now markup point lists and markup curves are supported) will be written to a csv file, accounting for any scaling you choose to set.

Markups in 3D Slicer are given in "anatomical" / "world" coordinates (i.e. relative to some world origin defined by the imager). This tool will convert them into SI units, then scale them per the AMBF scale that is set, and finally convert them into the LPS coordinate convention that is used in AMBF. These values will again be in "anatomical" coordinates, i.e. relative to the "[volume_name]_anatomical_origin" body in the ADF file.

NOTE: You must make your markups at the volume's *original* location, not on the labelmap after it was moved to an inital AMBF pose

## Example Usage of features
- Pick a volume of your choice, and make some segmentation on it (e.g. segment out bone from a CT volume)
- Convert the segmentation to a labelmap
- [Optional]: Change AMBF_Pose transform to show the labelmap volume in the position you would like relative to the AMBF origin
- Use this module to export the labelmap to a folder of png files, with a yaml file for the AMBF Simulator
- [Optional]: Draw a curve on the volume, and export it to a csv file
- Load the volume into the AMBF simulator (e.g. using the volumetric-drilling-plugin). If you generated a curve on the volume, use the ambf_trace_plugin (https://github.com/htp2/ambf_trace_plugin) to load the curve into the AMBF simulator using ```--csv_filename_static_traces <path_to_csv_file>``` and ```--static_trace_rel_body_name <volume_name>_anatomical_origin```


## Setup Instructions
After cloning this module:
- Open Slicer
- In the 'Welcome to Slicer' Module, click 'Customize Slicer'
- Press the Modules tab
- Under 'Paths' select 'Add'
- Select folder that contains this README document
- Restart Slicer as prompted
- The module should appear under the group name ```AMBF```
