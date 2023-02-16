# AMBF_Utils Module for 3D Slicer
This module is a collection of utilities for exporting volumes and markers from 3D Slicer to a format that can be read by the AMBF Simulator and various volumetric plugins for the AMBF Simulator.

Primarily, this module is expected to be used with continuum-manipulator-volumetric-drilling-plugin (https://github.com/htp2/continuum-manip-volumetric-drilling-plugin.git) and volumetric-drilling-plugin (https://github.com/LCSR-SICKKIDS/volumetric_drilling)

I would be remiss to not mention the work of Adnan Munawar et al., whose scripts in the referenced volumetric-drilling-plugin were used as a starting point for this module.

## Exporting LabelMaps (Slicer Segmentations) To AMBF Volumes:
Volumes will be exported into a folder of png files sliced in the superior/inferior plane. Their origin will be at the center of the volume, and the orientation will be LPS corresponding to x,y,z in its local coordinate system in AMBF. 

To assist with understanding coordinate systems in 3DSlicer, see the following link: https://www.slicer.org/wiki/Coordinate_systems

You may export a volume as a "grayscale" image, or it will be exported using the current color map of the segmentation.

The "volume" must be a labelmap (which you can create by segmenting a volume in 3D Slicer). You should enable visualization in the 3D viewer using the "Volume Rendering" Module to assist you in seeing what is going on. 

### Automatically Generate ADF files for AMBF Simulator
You can generate the accompanying AMBF ADF yaml file by checking the "Generate AMBF yaml" checkbox. This will generate a yaml file with the same name as the volume in the output directory which will have the size information, etc. setup so that the AMBF Simulator can read the volume.

In addition to the volume, it will also contain a body with [volume_name]_anatomical_origin which will be located at the anatomical or "space" origin of the volume. This is the origin applied to the image by the imager. Usually it is located somewhere outside the image itself. For example, in 3D Slicer your markup coordinates are defined relative to this anatomical origin. The only difference is that the anatomical origin will be defined using the LPS = (x,y,z) convention in AMBF, whereas it is RAS = (x,y,z) in 3D slicer.

### Applying a non-identity offset to the volume
When you select a LabelMap, a transform called "AMBF_Pose" will be generated in the "Transforms" module. You can move this around, and it will be updated in the ADF file. This is how you can specify your volume to have a non-identity pose. Both the volume's position and the anatomical origin's position will be updated in the ADF file accordingly. 

### A Note on the [volume_name]_anatomical_origin body
By default, this body will not move in AMBF. If you plan on moving the volume at any time in your application, you need to be sure to also move this body accordingly, otherwise you will no longer be able to use the anatomical origin as a reference point. By default, a visualization of the AMBF origin frame is shown (you can toggle this off). The relative pose in the 3D viewer to this origin frame will match the pose of the volume in the AMBF simulator to the AMBF origin. Please note: 3D slicer internally uses the RAS convention, whereas AMBF uses the LPS convention, so the transformation matrix will look different in 3D slicer for this reason. You can convert transforms from RAS to LPS using:
```python
# transforms are in hogomeneous 4x4 coordinates
ras2lps = np.diag([-1, -1, 1, 1])
transform_lps = ras2lps @ transform_ras @ ras2lps
```
To convert a position from RAS to LPS, you can simply negate the x and y coordinates.

## Exporting Markups
Markups (for now markup fiducials and markup curves are supported) will be to a csv file. A reference volume must be selected, and the markups will be exported in the coordinate system of the reference volume once loaded into AMBF. They will be in LPS = (x,y,z) convention relative to the anatomical origin of the volume (i.e. if you generated the volume with the "Generate AMBF yaml" checkbox checked, then the markup points will be relative to "[volume_name]_anatomical_origin" body in the ADF file).

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
