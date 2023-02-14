# create a plugin for 3DSlicer in a similar format to CM_path.py in this directory that can be used to convert
# Volumes into png slices, and convert markup points into a csv files that are in the correct frame of reference
# for the AMBF simulator.  This will allow us to use 3DSlicer to create the scene and then export it to AMBF.

import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import os
from Resources.slicer_helper import slicer_helper as sh
import numpy as np
import PIL.Image as Image
import PIL.Image
import numpy as np

#
# AMBF_utils
#

class AMBF_utils(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AMBF_utils"
        self.parent.categories = ["AMBF"]
        self.parent.dependencies = []
        self.parent.contributors = ["Henry Phalen (JHU)"]   
        self.parent.helpText = """
    This is a module to help with the AMBF simulator.
    """
        self.parent.acknowledgementText = """
    This file was originally developed by Henry Phalen, JHU.
    """

#
# AMBF_utilsWidget
#

class AMBF_utilsWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.dir = os.path.dirname(__file__)

        # Initialize Useful Parameters
        self.logic = AMBF_utilsLogic()
        self.logic.setup()

        # Setup main layout tabs (collapsible buttons)
        actionsCollapsibleButton = ctk.ctkCollapsibleButton()
        actionsCollapsibleButton.text = "Actions"
        self.layout.addWidget(actionsCollapsibleButton)
        actionsFormLayout = qt.QFormLayout(actionsCollapsibleButton)
        
        # output directory selector
        self.outputDirSelector = ctk.ctkPathLineEdit()
        self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        # set default path to tmp directory
        self.outputDirSelector.setCurrentPath(os.path.join(os.path.expanduser('~'), 'ambf_util_out'))
        self.outputDirSelector.setToolTip("Pick the output directory.")
        actionsFormLayout.addRow("Output Directory: ", self.outputDirSelector)

        # volume conversion tab
        volumeConversionCollapsibleButton = ctk.ctkCollapsibleButton()
        volumeConversionCollapsibleButton.text = "Volume Conversion"
        self.layout.addWidget(volumeConversionCollapsibleButton)
        volumeConversionFormLayout = qt.QFormLayout(volumeConversionCollapsibleButton)

        # segment labelmap selector
        self.segmentLabelMapSelector = slicer.qMRMLNodeComboBox()
        self.segmentLabelMapSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.segmentLabelMapSelector.selectNodeUponCreation = True
        self.segmentLabelMapSelector.addEnabled = False
        self.segmentLabelMapSelector.removeEnabled = False
        self.segmentLabelMapSelector.noneEnabled = False
        self.segmentLabelMapSelector.showHidden = False
        self.segmentLabelMapSelector.showChildNodeTypes = False
        self.segmentLabelMapSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentLabelMapSelector.setToolTip("Pick the labelmap to export.")
        volumeConversionFormLayout.addRow("LabelMap: ", self.segmentLabelMapSelector)

        # checkbox for whether to export the labelmap as a grayscale image
        self.exportLabelMapAsGrayscale = qt.QCheckBox()
        self.exportLabelMapAsGrayscale.checked = False
        self.exportLabelMapAsGrayscale.setToolTip("Export the as a grayscale images, else uses current color")
        volumeConversionFormLayout.addRow("Export as Grayscale: ", self.exportLabelMapAsGrayscale)

        # image prefix
        self.imagePrefix = qt.QLineEdit()
        self.imagePrefix.text = "plane000"
        self.imagePrefix.setToolTip("Prefix for the image files")
        volumeConversionFormLayout.addRow("Image Prefix: ", self.imagePrefix)

        # checkbox (default checked) for generating AMBF yaml file, next to this a text field for the name of volume that is not enabled if the checkbox is not checked
        self.generateYaml = qt.QCheckBox()
        self.generateYaml.checked = True
        self.generateYaml.setToolTip("Generate a yaml file for AMBF")
        volumeConversionFormLayout.addRow("Generate AMBF yaml: ", self.generateYaml)

        # text field for the name of the volume to be used in the yaml file
        self.volumeName = qt.QLineEdit()
        self.volumeName.text = "volume"
        self.volumeName.setToolTip("Name of the volume to be used in the yaml file")
        volumeConversionFormLayout.addRow("AMBF Volume Name: ", self.volumeName)

        # ambf scale
        self.ambfScale = qt.QDoubleSpinBox()
        self.ambfScale.value = 1.0
        self.ambfScale.setToolTip("Scale the volume by this factor")
        volumeConversionFormLayout.addRow("AMBF Scale: ", self.ambfScale)
        

        # volumeName is disabled if the checkbox is not checked
        self.volumeName.enabled = self.generateYaml.checked
        self.ambfScale.enabled = self.generateYaml.checked
        # connect the checkbox to the volumeName to enable/disable it
        self.generateYaml.connect('stateChanged(int)', self.onGenerateYamlCheckbox)
        # disable the volumeName if the checkbox is not checked

        # Create a button to export the LabelMap Node to png slices
        self.exportLabelMapButton = qt.QPushButton("Export LabelMap to PNG")
        self.exportLabelMapButton.toolTip = "Export the labelmap to png slices."
        self.exportLabelMapButton.enabled = True
        volumeConversionFormLayout.addRow(self.exportLabelMapButton)
        self.exportLabelMapButton.connect('clicked(bool)', self.onExportLabelMapButton)
        
        # markup conversion tab
        markupConversionCollapsibleButton = ctk.ctkCollapsibleButton()
        markupConversionCollapsibleButton.text = "Markup Conversion"
        self.layout.addWidget(markupConversionCollapsibleButton)
        markupConversionFormLayout = qt.QFormLayout(markupConversionCollapsibleButton)

        # reference volume selector
        self.referenceVolumeSelector = slicer.qMRMLNodeComboBox()
        self.referenceVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.referenceVolumeSelector.selectNodeUponCreation = True
        self.referenceVolumeSelector.addEnabled = False
        self.referenceVolumeSelector.removeEnabled = False
        self.referenceVolumeSelector.noneEnabled = False
        self.referenceVolumeSelector.showHidden = False
        self.referenceVolumeSelector.showChildNodeTypes = False
        self.referenceVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.referenceVolumeSelector.setToolTip("Pick the reference volume.")
        markupConversionFormLayout.addRow("Reference Volume: ", self.referenceVolumeSelector)

        # markup selector
        self.markupSelector = slicer.qMRMLNodeComboBox()
        self.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode", "vtkMRMLMarkupsCurveNode"]
        self.markupSelector.selectNodeUponCreation = True
        self.markupSelector.addEnabled = False
        self.markupSelector.removeEnabled = False
        self.markupSelector.noneEnabled = False
        self.markupSelector.showHidden = False
        self.markupSelector.showChildNodeTypes = False
        self.markupSelector.setMRMLScene(slicer.mrmlScene)
        self.markupSelector.setToolTip("Pick the markup to export.")
        markupConversionFormLayout.addRow("Markup: ", self.markupSelector)

        # output name selector
        self.outputName = qt.QLineEdit()
        self.outputName.text = "markup.csv"
        self.outputName.setToolTip("Name of the output file")
        markupConversionFormLayout.addRow("Output Name: ", self.outputName)
        # have name automatically update to the name of the markup node when it is changed
        self.markupSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onMarkupNodeChanged)

        # Create a button to export the markup points to csv
        self.exportMarkupButton = qt.QPushButton("Export Markup Points to CSV for AMBF")
        self.exportMarkupButton.toolTip = "Export the markup points to csv."
        self.exportMarkupButton.enabled = True
        markupConversionFormLayout.addRow(self.exportMarkupButton)
        self.exportMarkupButton.connect('clicked(bool)', self.onExportMarkupButton)       

        # Add vertical spacer
        self.layout.addStretch(1)

    def cleanup(self):
        pass

    def onExportLabelMapButton(self):
        self.logic.exportLabelMapToPNG(self.segmentLabelMapSelector.currentNode(), self.outputDirSelector.currentPath, 
            self.imagePrefix.text, self.exportLabelMapAsGrayscale.checked, 
            self.generateYaml.checked, self.volumeName.text, self.ambfScale.value)

    def onExportMarkupButton(self):
        self.logic.exportMarkupToCSV(self.markupSelector.currentNode(), self.referenceVolumeSelector.currentNode(),
            self.outputDirSelector.currentPath, self.outputName.text, self.ambfScale.value)

    def onGenerateYamlCheckbox(self):
        self.volumeName.enabled = self.generateYaml.checked
        self.ambfScale.enabled = self.generateYaml.checked
    
    def onMarkupNodeChanged(self):
        self.outputName.text = self.markupSelector.currentNode().GetName()+".csv"
    

#
# AMBF_utilsLogic
#

class AMBF_utilsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
    def __init__(self):
        pass

    def setup(self):
        pass

    def exportLabelMapToPNG(self, labelMapNode, outputDir, image_prefix, grayscale, generateYaml, volume_name, scale):
        if labelMapNode is None:
            logging.error("Segmentation node is None")
            return

        if not os.path.exists(outputDir):
            logging.error("Output directory does not exist")
            return

        # get the labelmap from the labelmap node
        labelMap = labelMapNode.GetImageData()

        yaml_save_location = outputDir

        # we will fill a directory with png slices
        slice_dir = os.path.join(outputDir, volume_name)
        if not os.path.exists(slice_dir):
            os.mkdir(slice_dir)
        
        # get the dimensions of the labelmap
        dimensions = labelMap.GetDimensions()

        # get the spacing of the labelmap
        spacing = labelMap.GetSpacing()

        # get the origin of the labelmap
        origin = labelMap.GetOrigin()

        # get the number of components in the labelmap
        numberOfComponents = labelMap.GetNumberOfScalarComponents()

        # get the number of points in the image data
        numberOfPoints = labelMap.GetNumberOfPoints()
        # get the number of pixels in the image data
        numberOfPixels = int(numberOfPoints / numberOfComponents)
        # get the number of slices in the image data
        numberOfSlices = int(numberOfPixels / (dimensions[0] * dimensions[1]))
        print("Number of slices: " + str(numberOfSlices))
        
        # get the pixel data from the image data
        pixelData = labelMap.GetPointData().GetScalars()

        # get the pixel data as a numpy array
        pixelDataArray = vtk.util.numpy_support.vtk_to_numpy(pixelData)

        # reshape the pixel data array to a 3D array, this order maintains correct shape, but results in (x,y,z) = (S,A,R) with origin at top left corner
        # L/R is left, right, P/A is posterior, anterior, S/I is superior, inferior, positive is towards the name, i.e. LPS means +x, +y, +z are towards the left, posterior, superior
        pixelDataArray3D = pixelDataArray.reshape((dimensions[2], dimensions[1], dimensions[0]))

        # now, we want to rearrange the dimensions so that we arrive at (x,y,z) = (L,P,S) as read into AMBF
        # currently, we have (x,y,z) = (S,A,R) with the origin at the top left corner
        # ambf will read in a volume as 2D image slices by increasing slice order. These individual images are read in (W,H) from the bottom left corner
        # the array here however has origin at the top left corner, so (array_x, array_y, array_z) will be read in as (array_y, -array_x, array_z)
        # where the negative sign means a flip in direction

        # so if want (L,P,S) to be read, we need to input (P,-L,S) = (P,R,S)
        pixelDataArray3D = np.swapaxes(pixelDataArray3D, 0, 2) #(S,A,R) --> (R,A,S)
        pixelDataArray3D = np.swapaxes(pixelDataArray3D, 0, 1) #(R,A,S) --> (A,R,S)
        pixelDataArray3D = np.flip(pixelDataArray3D, 0) #(A,R,S) --> (P,R,S)

        data_size = pixelDataArray3D.shape
        # dimensions are in terms of RAS, we will be using LPS but that doesn't change the dimensions which are not signed
        dimensions_mm = np.array(dimensions)
        dimensions_m = 0.001*(dimensions_mm)

        # the origin tells us what the "anatomical" position of the [0,0,0] voxel is in mm. 
        # 3D slicer defines the origin as the bottom left corner of the volume, but AMBF defines it as the center)
        # [TODO: account for change in AMBF origin to bottom left corner if that occurs]
        origin_mm = scale*(origin + (dimensions_mm/2))
        origin_m = 0.001 * origin_mm

        if grayscale:
            normalized_data = self.normalize_data(pixelDataArray3D)
            scaled_data = self.scale_data(normalized_data, 255.9)
            self.save_volume_as_images(scaled_data, os.path.join(slice_dir, image_prefix))
        else: # return color image using the active color node / color table
            colorNode = labelMapNode.GetDisplayNode().GetColorNode()

            for i in range(pixelDataArray3D.shape[2]):
                #make rbga image with size of pixelDataArray3d.shape[0] and pixelDataArray3d.shape[1]
                im_name = image_prefix + str(i) + '.png'
                img = np.zeros((pixelDataArray3D.shape[0], pixelDataArray3D.shape[1], 4))
                # find each unique value in the slice
                unique_values = np.unique(pixelDataArray3D[:,:,i])
                # for each unique value, find the color and set the pixel to that color
                for j in unique_values:
                    color = np.array([0.0,0.0,0.0,0.0])
                    colorNode.GetColor(j, color)
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
                    # set the pixel to the color if the scalar value is j
                    img[pixelDataArray3D[:,:,i] == j] = color
                # convert to PIL RBGA image
                img = PIL.Image.fromarray(np.uint8(img))

                img.save(os.path.join(slice_dir, im_name))

        if generateYaml:
            print("data_size: " + str(data_size))
            self.save_yaml_file(data_size, dimensions_m, volume_name, yaml_save_location, origin_m, scale, image_prefix)


    def convert_png_transparent(self, image, bg_color=(255,255,255)):
        # https://stackoverflow.com/questions/765736/how-to-use-pil-to-make-all-white-pixels-transparent
        # Jonathan Dauwe
        array = np.array(image, dtype=np.ubyte)
        mask = (array[:,:,:3] == bg_color).all(axis=2)
        alpha = np.where(mask, 0, 255)
        array[:,:,-1] = alpha
        return PIL.Image.fromarray(np.ubyte(array))
        
    def save_image(self, array, im_name):
        img = PIL.Image.fromarray(array.astype(np.uint8))
        img = img.convert("RGBA")
        img = self.convert_png_transparent(img, bg_color=(0,0,0))
        img.save(im_name)


    def normalize_data(self, data):
        max = data.max()
        min = data.min()
        if max==min:
            if min!= 0: # assume entire image is single volume
                normalized_data = data/min
            # here, else is implicit - image is all zero and will remain that way
        else:
            normalized_data = (data - min) / float(max - min)
        return normalized_data


    def scale_data(self, data, scale):
        scaled_data = data * scale
        return scaled_data


    def save_volume_as_images(self, data, im_prefix):
        for i in range(data.shape[2]):
            im_name = im_prefix + str(i) + '.png'
            self.save_image(data[:, :, i], im_name)

    def save_yaml_file(self, data_size, dimensions, volume_name, yaml_save_location, origin, scale, prefix):
        lines = []
        lines.append("# AMBF Version: (0.1)")
        lines.append("bodies: []")
        lines.append("joints: []")
        lines.append("volumes: [VOLUME "+volume_name+"]")
        lines.append("high resolution path: ./meshes/high_res/")
        lines.append("low resolution path: ./meshes/low_res/")
        lines.append("ignore inter-collision: true")
        lines.append("namespace: /ambf/env/")
        lines.append("")
        lines.append("VOLUME "+volume_name+":")
        lines.append("  name: "+volume_name)
        lines.append("  location:")
        lines.append("    position: {x: " + str(origin[0])+", y: "+str(origin[1])+", z: "+str(origin[2])+"}")
        lines.append("    orientation: {r: 0.0, p: 0.0, y: 0.0}")
        lines.append("  scale: "+str(scale))
        lines.append("  dimensions: {x: "+str(dimensions[0])+", y: "+str(dimensions[1])+", z: " + str(dimensions[2]) +"}")
        lines.append("  images:")
        lines.append("    path: ../resources/volumes/"+volume_name+"/")
        lines.append("    prefix: "+prefix)
        lines.append("    format: png")
        lines.append("    count: " + str(max(data_size)))  # Note this can be larger than actual value
        lines.append("  shaders:")
        lines.append("    path: ./shaders/volume/")
        lines.append("    vertex: shader.vs")
        lines.append("    fragment: shader.fs")
        yaml_name = os.path.join(yaml_save_location, volume_name+".yaml")
        with open(yaml_name, 'w') as f:
            f.write('\n'.join(lines))
            f.close()
        print("Saved YAML file to: " + yaml_name)

    def exportMarkupToCSV(self, markupNode, volumeNode, output_dir, output_name, ambf_scale=1.0):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get the number of control points in the markup
        numPoints = markupNode.GetNumberOfControlPoints()
        print("Number of control points: " + str(numPoints))

        slicer_dim_to_m = 1.0 / 1000.0
        m_to_ambf_dim = ambf_scale

        # get volume origin and spacing, size, spacing as numpy array
        origin_slicer = np.array(volumeNode.GetOrigin())
        # origin ras to lps
        origin_slicer[0] = -origin_slicer[0]
        origin_slicer[1] = -origin_slicer[1]

        spacing_slicer = np.array(volumeNode.GetSpacing())
        size_voxels = np.array(volumeNode.GetImageData().GetDimensions())

        # convert to SI units
        origin_m = origin_slicer * slicer_dim_to_m
        spacing_m = spacing_slicer * slicer_dim_to_m
        size_m = size_voxels * spacing_m
        print("Volume origin (m): " + str(origin_m))
        print("Volume spacing (m): " + str(spacing_m))
        print("Volume size (m): " + str(size_m))

        # ambf origin is in center of volume, not corner
        ambf_p_slicer = origin_m + (size_m/2)
        # ambf coordinates are rotated by pi/2 about z axis
        ambf_R_slicer = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        ambf_T_slicer = np.eye(4)
        ambf_T_slicer[:3, :3] = ambf_R_slicer
        ambf_T_slicer[:3, 3] = ambf_p_slicer

        # get the markup points
        markupPoints = []
        for i in range(numPoints):
            markupPoints.append(markupNode.GetNthControlPointPosition(i))
        markupPoints = np.array(markupPoints)
        # ras to lps
        markupPoints[:, 0] = -markupPoints[:, 0]
        markupPoints[:, 1] = -markupPoints[:, 1]

        # convert to SI units for transformation
        markupPoints = markupPoints * slicer_dim_to_m

        # convert to ambf coordinates by multiplying each point by the transformation matrix
        ambf_p_markup = np.dot(np.linalg.inv(ambf_T_slicer), np.hstack((markupPoints, np.ones((numPoints, 1)))).T).T
        ambf_p_markup = ambf_p_markup[:, :3]

        # convert units to ambf_dimensions
        ambf_p_markup = ambf_p_markup * m_to_ambf_dim

        # save to csv file
        # check if outputname has .csv extension, add it if not
        if not output_name.endswith(".csv"):
            output_name = output_name + ".csv"
        csv_name = os.path.join(output_dir, output_name)
        np.savetxt(csv_name, ambf_p_markup, delimiter=",")


#
# AMBF_utilsTest
#

class AMBF_utilsTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.dummy_test()

    def dummy_test(self):
        self.delayDisplay("Starting the test:")
        self.delayDisplay("No tests for now!")
        self.delayDisplay('Test passed!')