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
import numpy as np
import PIL.Image as Image
import PIL.Image
import numpy as np
from scipy.spatial.transform import Rotation

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

        # checkbox for Show AMBF Origin
        self.AMBF_X = slicer.vtkMRMLMarkupsLineNode()
        self.AMBF_X.SetName("AMBF_X")
        self.AMBF_X.GetMeasurement("length").SetEnabled(0)
        slicer.mrmlScene.AddNode(self.AMBF_X)
        self.AMBF_Y = slicer.vtkMRMLMarkupsLineNode()
        self.AMBF_Y.SetName("AMBF_Y")
        self.AMBF_Y.GetMeasurement("length").SetEnabled(0)
        slicer.mrmlScene.AddNode(self.AMBF_Y)
        self.AMBF_Z = slicer.vtkMRMLMarkupsLineNode()
        self.AMBF_Z.SetName("AMBF_Z")
        self.AMBF_Z.GetMeasurement("length").SetEnabled(0)
        slicer.mrmlScene.AddNode(self.AMBF_Z)

        self.showAmbfOrigin = qt.QCheckBox()
        self.showAmbfOrigin.checked = True
        self.showAmbfOrigin.setToolTip("Show the AMBF origin")
        actionsFormLayout.addRow("Show AMBF Origin: ", self.showAmbfOrigin)
        # connect the checkbox to the function that shows/hides the AMBF origin
        self.showAmbfOrigin.connect('stateChanged(int)', self.onShowAmbfOriginChanged)


        # volume conversion tab
        volumeConversionCollapsibleButton = ctk.ctkCollapsibleButton()
        volumeConversionCollapsibleButton.text = "Segmentation to AMBF Volume"
        self.layout.addWidget(volumeConversionCollapsibleButton)
        volumeConversionFormLayout = qt.QFormLayout(volumeConversionCollapsibleButton)

        # segment labelmap selector
        self.segmentLabelMapSelector = slicer.qMRMLNodeComboBox()
        self.segmentLabelMapSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.segmentLabelMapSelector.selectNodeUponCreation = False
        self.segmentLabelMapSelector.addEnabled = True
        self.segmentLabelMapSelector.removeEnabled = False
        self.segmentLabelMapSelector.noneEnabled = True
        self.segmentLabelMapSelector.showHidden = False
        self.segmentLabelMapSelector.showChildNodeTypes = False
        self.segmentLabelMapSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentLabelMapSelector.setToolTip("Pick the labelmap to export.")
        volumeConversionFormLayout.addRow("LabelMap: ", self.segmentLabelMapSelector)
        self.segmentLabelMapSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSegmentLabelMapChanged)

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

        # checkbox (default checked) for generating image slices
        self.generateImageSlices = qt.QCheckBox()
        self.generateImageSlices.checked = True
        self.generateImageSlices.setToolTip("Generate image slices")
        volumeConversionFormLayout.addRow("Generate Image Slices: ", self.generateImageSlices)


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
        
        # add text box that says "You can change the pose of the volume in AMBF by changing the Transform AMBF_Pose"
        self.poseText = qt.QLabel("**You can change the pose of the volume in AMBF by changing the Transform \"AMBF_Pose\"**")
        # allow the text to wrap
        self.poseText.wordWrap = True
        volumeConversionFormLayout.addRow(self.poseText)

        # add a checkbox to enable/disable labelmap volume rendering
        self.enableLabelMapVolumeRendering = qt.QCheckBox()
        self.enableLabelMapVolumeRendering.checked = True
        self.enableLabelMapVolumeRendering.setToolTip("Enable volume rendering of the labelmap")
        volumeConversionFormLayout.addRow("Enable LabelMap Volume Rendering: ", self.enableLabelMapVolumeRendering)
        # connect the checkbox to the function that enables/disables volume rendering
        self.enableLabelMapVolumeRendering.connect('stateChanged(int)', self.onEnableLabelMapVolumeRenderingChanged)

        # add a checkbox to enable/disable labelmap rendering at AMBFPose
        self.enableLabelMapRenderingAtAmbfPose = qt.QCheckBox()
        self.enableLabelMapRenderingAtAmbfPose.checked = True
        self.enableLabelMapRenderingAtAmbfPose.setToolTip("Enable rendering of the labelmap at the AMBF pose")
        volumeConversionFormLayout.addRow("Show LabelMap Rendering at AMBF Pose: ", self.enableLabelMapRenderingAtAmbfPose)
        # connect the checkbox to the function that enables/disables rendering at AMBFPose
        self.enableLabelMapRenderingAtAmbfPose.connect('stateChanged(int)', self.onEnableLabelMapRenderingAtAmbfPoseChanged)

        # volumeName is disabled if the checkbox is not checked
        self.volumeName.enabled = self.generateYaml.checked
        self.ambfScale.enabled = self.generateYaml.checked
        # connect the checkbox to the volumeName to enable/disable it
        self.generateYaml.connect('stateChanged(int)', self.onGenerateYamlCheckbox)
        
        # Create a button to export the LabelMap Node to png slices
        self.exportLabelMapButton = qt.QPushButton("Export LabelMap to PNGs for AMBF")
        self.exportLabelMapButton.toolTip = "Export the labelmap to png slices."
        self.exportLabelMapButton.enabled = True
        volumeConversionFormLayout.addRow(self.exportLabelMapButton)
        self.exportLabelMapButton.connect('clicked(bool)', self.onExportLabelMapButton)
        
        # Add a tab called "yaml output"
        yamlOutputCollapsibleButton = ctk.ctkCollapsibleButton()
        yamlOutputCollapsibleButton.text = "YAML Output"
        self.layout.addWidget(yamlOutputCollapsibleButton)
        yamlOutputFormLayout = qt.QFormLayout(yamlOutputCollapsibleButton)
        # start collapsed
        yamlOutputCollapsibleButton.collapsed = True

        # print the current yaml output to the screen for the user to look at and edit if they want
        self.yamlOutput = qt.QTextEdit()
        self.yamlOutput.setReadOnly(False)
        yamlOutputFormLayout.addRow(self.yamlOutput)

        # use qhboxlayout to put the two buttons next to each other
        hbox = qt.QHBoxLayout()
        yamlOutputFormLayout.addRow(hbox)

        # add a button to refresh the text box with the contents of the yaml file
        self.refreshYamlButton = qt.QPushButton("Load/Refresh YAML File")
        self.refreshYamlButton.toolTip = "Load the yaml file into the text box."
        self.refreshYamlButton.enabled = True
        hbox.addWidget(self.refreshYamlButton)
        self.refreshYamlButton.connect('clicked(bool)', self.onRefreshYamlButton)

        # add a button to overwrite the yaml file with the contents of the text box
        self.overwriteYamlButton = qt.QPushButton("Overwrite YAML File")
        self.overwriteYamlButton.toolTip = "Overwrite the yaml file with the contents of the text box."
        self.overwriteYamlButton.enabled = True
        hbox.addWidget(self.overwriteYamlButton)
        self.overwriteYamlButton.connect('clicked(bool)', self.onOverwriteYamlButton)

        # markup conversion tab
        markupConversionCollapsibleButton = ctk.ctkCollapsibleButton()
        markupConversionCollapsibleButton.text = "Markup Conversion"
        self.layout.addWidget(markupConversionCollapsibleButton)
        markupConversionFormLayout = qt.QFormLayout(markupConversionCollapsibleButton)

        # reference volume selector
        self.referenceVolumeSelector = slicer.qMRMLNodeComboBox()
        self.referenceVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"]
        self.referenceVolumeSelector.selectNodeUponCreation = False
        self.referenceVolumeSelector.addEnabled = True
        self.referenceVolumeSelector.removeEnabled = False
        self.referenceVolumeSelector.noneEnabled = True
        self.referenceVolumeSelector.showHidden = False
        self.referenceVolumeSelector.showChildNodeTypes = False
        self.referenceVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.referenceVolumeSelector.setToolTip("Pick the reference volume.")
        markupConversionFormLayout.addRow("Reference Volume: ", self.referenceVolumeSelector)

        # markup selector
        self.markupSelector = slicer.qMRMLNodeComboBox()
        self.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode", "vtkMRMLMarkupsCurveNode"]
        self.markupSelector.selectNodeUponCreation = True
        self.markupSelector.addEnabled = True
        self.markupSelector.removeEnabled = False
        self.markupSelector.noneEnabled = False
        self.markupSelector.showHidden = False
        self.markupSelector.showChildNodeTypes = False
        self.markupSelector.setMRMLScene(slicer.mrmlScene)
        self.markupSelector.setToolTip("Pick the markup to export.")
        markupConversionFormLayout.addRow("Markup: ", self.markupSelector)

        # set this node to always contain the transform from the current volume origin to its center point
        self.referenceVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onReferenceVolumeChanged)

        self.ambfPose = slicer.vtkMRMLLinearTransformNode()
        self.ambfPose.SetName("AMBF_Pose")
        slicer.mrmlScene.AddNode(self.ambfPose)

        # create two transform nodes called RASTOLPS1 and RASTOLPS2 that have rotation matrices diag[-1,-1,1]
        self.RASTOLPS = slicer.vtkMRMLLinearTransformNode()
        self.RASTOLPS.SetName("RASToLPS")
        slicer.mrmlScene.AddNode(self.RASTOLPS)
        self.RASTOLPS2 = slicer.vtkMRMLLinearTransformNode()
        self.RASTOLPS2.SetName("RASToLPS2")
        slicer.mrmlScene.AddNode(self.RASTOLPS2)
        ras2lps = np.eye(4)
        ras2lps[0,0] = -1
        ras2lps[1,1] = -1
        slicer.util.updateTransformMatrixFromArray(self.RASTOLPS, ras2lps)
        slicer.util.updateTransformMatrixFromArray(self.RASTOLPS2, ras2lps)

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
            self.generateYaml.checked, self.volumeName.text, self.ambfScale.value, self.ambfPose, self.generateImageSlices.checked)
        self.onRefreshYamlButton()

    def onExportMarkupButton(self):
        self.logic.exportMarkupToCSV(self.markupSelector.currentNode(), self.referenceVolumeSelector.currentNode(),
            self.outputDirSelector.currentPath, self.outputName.text, self.ambfScale.value)

    def onGenerateYamlCheckbox(self):
        self.volumeName.enabled = self.generateYaml.checked
        self.ambfScale.enabled = self.generateYaml.checked        

    def onMarkupNodeChanged(self):
        self.outputName.text = self.markupSelector.currentNode().GetName()+".csv"

    def onReferenceVolumeChanged(self):
        pass
    
    def onOverwriteYamlButton(self):
        with open(self.outputDirSelector.currentPath+"/"+self.volumeName.text+".yaml", "w") as f:
            f.write(self.yamlOutput.toPlainText())
    

    def onSegmentLabelMapChanged(self):
        # set anatomical_T_AMBF to the transform from the current volume origin to its center point
        volumeNode = self.segmentLabelMapSelector.currentNode()
        if volumeNode is None:
            return

        # get volume rendering node
        volRenLogic = slicer.modules.volumerendering.logic()
        self.volren_displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)

        # We have decided that if you use the identity transform, space_origin / antatomical_origin will be placed at the AMBF origin, with LPS -> x,y,z 
        self.update_AMBF_axes(np.array((0.0, 0.0, 0.0))) # assumes the anatomical_origin is (0,0,0) in 3D slicer
        self.updateTransformRelations()
        self.onEnableLabelMapVolumeRenderingChanged()

    def onShowAmbfOriginChanged(self):
        self.AMBF_X.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)
        self.AMBF_Y.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)
        self.AMBF_Z.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)

    def update_AMBF_axes(self, origin):
        self.AMBF_X.SetLineStartPosition(origin)
        self.AMBF_X.SetLineEndPosition(origin - np.array([100.0,0.0,0.0])) # minus because it will be LPS
        self.AMBF_X.GetDisplayNode().SetColor(1.0,0.0,0.0)
        self.AMBF_X.GetDisplayNode().SetSelectedColor(1.0,0.0,0.0)
        self.AMBF_X.GetDisplayNode().SetLineThickness(1.0)
        self.AMBF_X.SetSelectable(0)
        self.AMBF_X.SetLocked(1)
        self.AMBF_X.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)

        self.AMBF_Y.SetLineStartPosition(origin)
        self.AMBF_Y.SetLineEndPosition(origin - np.array([0.0,100.0,0.0])) # minus because it will be LPS
        self.AMBF_Y.GetDisplayNode().SetColor(0.0,1.0,0.0)
        self.AMBF_Y.GetDisplayNode().SetSelectedColor(0.0,1.0,0.0)
        self.AMBF_Y.GetDisplayNode().SetLineThickness(1.0)
        self.AMBF_Y.SetSelectable(0)
        self.AMBF_Y.SetLocked(1)
        self.AMBF_Y.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)

        self.AMBF_Z.SetLineStartPosition(origin)
        self.AMBF_Z.SetLineEndPosition(origin + np.array([0.0,0.0,100.0])) # plus because it will be LPS
        self.AMBF_Z.GetDisplayNode().SetColor(0.0,0.0,1.0)
        self.AMBF_Z.GetDisplayNode().SetSelectedColor(0.0,0.0,1.0)
        self.AMBF_Z.GetDisplayNode().SetLineThickness(1.0)
        self.AMBF_Z.SetSelectable(0)
        self.AMBF_Z.SetLocked(1)
        self.AMBF_Z.GetDisplayNode().SetVisibility(self.showAmbfOrigin.checked)
    


    def updateTransformRelations(self):        
        # These intermediate transforms are necessary because we want AMBF_Pose to act
        # as if it were occuring on the anatomical_origin in LPS coordinates. By default, in RAS coordinates.
        self.ambfPose.SetAndObserveTransformNodeID(self.RASTOLPS.GetID())
        self.RASTOLPS2.SetAndObserveTransformNodeID(self.ambfPose.GetID())

        labelMapNode = self.segmentLabelMapSelector.currentNode()
        if labelMapNode is not None:
            if self.enableLabelMapRenderingAtAmbfPose.checked:
                labelMapNode.SetAndObserveTransformNodeID(self.RASTOLPS2.GetID())
            else:
                labelMapNode.SetAndObserveTransformNodeID(None)


    def onRefreshYamlButton(self):
        # check if file exists
        filename = self.outputDirSelector.currentPath+"/"+self.volumeName.text+".yaml"
        if not os.path.isfile(filename):
            self.yamlOutput.setText("")
            return
        
        with open(self.outputDirSelector.currentPath+"/"+self.volumeName.text+".yaml", 'r') as stream:
            # check if file exists
            if stream is not None:
                txt = stream.read()
                self.yamlOutput.setText(txt)
            else:
                self.yamlOutput.setText("")

    def onEnableLabelMapVolumeRenderingChanged(self):
        if self.enableLabelMapVolumeRendering.checked:
            self.volren_displayNode.SetVisibility(1)
        else:
            self.volren_displayNode.SetVisibility(0)
    
    def onEnableLabelMapRenderingAtAmbfPoseChanged(self):
        self.updateTransformRelations()

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

    def calculate_VolumeOriginInAMBF_p_VolumeOriginInSlicer(self, volumeNode):
        # assumes AMBF is at volume center with x,y,z axes aligned with LPS
        # get volume center in RAS coordinates (some help from https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html)
        volumeArray = slicer.util.arrayFromVolume(volumeNode) # NOTE: this utility makes an array in K,J,I order, not I,J,K
        center_kji = np.array(volumeArray.shape)/2
        center_ijk = np.flip(center_kji)
        IJKToRASMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(IJKToRASMatrix)
        volumeCenter_p_spaceOrigin = np.array(IJKToRASMatrix.MultiplyPoint(np.append(center_ijk,1.0))[0:3])
        return volumeCenter_p_spaceOrigin

    def exportLabelMapToPNG(self, labelMapNode, yaml_save_location, image_prefix, grayscale, generateYaml, volume_name, scale, ambf_pose_node, generate_images):
        if labelMapNode is None:
            logging.error("Segmentation node is None")
            return

        if not os.path.exists(yaml_save_location):
            logging.error(f"Output directory {yaml_save_location} does not exist")
            return

        # get the labelmap from the labelmap node
        labelMap = labelMapNode.GetImageData()

        # get the dimensions of the labelmap (these are voxel dimensions)
        vox_dims_slicer = np.array(labelMap.GetDimensions())

        # get the spacing of the labelmap (these are mm per voxel in each dimension)
        spacing_mm_per_vox = np.array(labelMapNode.GetSpacing())

        # calculate the size of the labelmap (physical length of each dimension)
        logging.info("Voxel Dimensions: " + str(vox_dims_slicer))
        logging.info("Voxel Spacing (mm): " + str(spacing_mm_per_vox))
        size_mm = vox_dims_slicer * spacing_mm_per_vox
        size_m = size_mm / 1000.0

        # get the origin of the labelmap
        volume_origin = np.array(labelMapNode.GetOrigin())
        
        # get the pixel data from the image data
        pixelData = labelMap.GetPointData().GetScalars()

        # get the pixel data as a numpy array
        pixelDataArray = vtk.util.numpy_support.vtk_to_numpy(pixelData)

        # reshape the pixel data array to a 3D array, this order maintains correct shape, but results in (x,y,z) = (S,A,R) with origin at top left corner
        # L/R is left, right, P/A is posterior, anterior, S/I is superior, inferior, positive is towards the name, i.e. LPS means +x, +y, +z are towards the left, posterior, superior
        pixelDataArray3D = pixelDataArray.reshape((vox_dims_slicer[2], vox_dims_slicer[1], vox_dims_slicer[0]))

        # now, we want to rearrange the dimensions so that we arrive at (x,y,z) = (L,P,S) as read into AMBF
        # currently, we have (x,y,z) = (S,A,R) with the origin at the top left corner
        # ambf will read in a volume as 2D image slices by increasing slice order. These individual images are read in (W,H) from the bottom left corner
        # the array here however has origin at the top left corner, so (array_x, array_y, array_z) will be read in as (array_y, -array_x, array_z)
        # where the negative sign means a flip in direction

        # so if want (L,P,S) to be read, we need to input (P,-L,S) = (P,R,S)
        pixelDataArray3D = np.swapaxes(pixelDataArray3D, 0, 2) #(S,A,R) --> (R,A,S)
        pixelDataArray3D = np.swapaxes(pixelDataArray3D, 0, 1) #(R,A,S) --> (A,R,S)
        pixelDataArray3D = np.flip(pixelDataArray3D, 0) #(A,R,S) --> (P,R,S)

        vox_dims_ambf = np.array(pixelDataArray3D.shape)

        VolumeOriginInAMBF_p_VolumeOriginInSlicer = self.calculate_VolumeOriginInAMBF_p_VolumeOriginInSlicer(labelMapNode)
        VolumeOriginInAMBF_p_VolumeOriginInSlicer_m = VolumeOriginInAMBF_p_VolumeOriginInSlicer * 0.001
        
        # But that result was in RAS coordinates, we want to convert to LPS
        VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps = VolumeOriginInAMBF_p_VolumeOriginInSlicer_m
        VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps[0] = -VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps[0]
        VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps[1] = -VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps[1]
    
        if generate_images:
            # we will fill a directory with png slices
            slice_dir = os.path.join(yaml_save_location, volume_name)
            if not os.path.exists(slice_dir):
                os.mkdir(slice_dir)

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
            print("vox_dims_ambf: " + str(vox_dims_ambf))
            self.save_yaml_file(vox_dims_ambf, size_m, volume_name, yaml_save_location, VolumeOriginInAMBF_p_VolumeOriginInSlicer_m_lps, scale, image_prefix, ambf_pose_node)


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

    
    def save_yaml_file(self, data_size, dimensions, volume_name, yaml_save_location, origin_m, scale, prefix, ambf_pose_node):
        # data_size: the size of the volume in voxels: (x,y,z) corresponds to anatomical dimensions (l,p,s)
        # dimensions: the size of the volume in meters per dimension (x,y,z)
        # volume_name: the name of the volume in AMBF
        # yaml_save_location: the location to save the yaml file
        # origin_m: the relative translation of the ambf volume origin (center of the volume) to the anatomical origin
        # scale: the scale of the volume for AMBF (1.0 is metric)
        # prefix: the prefix of the image files # TODO: cleaner way to zero pad the names
        # ambf_pose_node: a node that defines the initial pose of the anatomical origin in AMBF

        data_size = np.array(data_size)
        # unpack ambf pose to numpy array
        ambf_pose = slicer.util.arrayFromTransformMatrix(ambf_pose_node)
        # convert to m
        ambf_pose_m = ambf_pose
        ambf_pose_m[0:3,3] = ambf_pose_m[0:3,3] * 0.001
        # due to the use of ras2lps transforms in the module, we are already in the lps convention
        ambf_pose_m_lps = ambf_pose_m

        # Determine where the anatomical_origin will be in AMBF. This is just offset by the "ambf_pose_m_lps" value
        anor_pos_x, anor_pos_y, anor_pos_z = ambf_pose_m_lps[0:3,3] * scale
        anor_rot_r,anor_rot_p,anor_rot_y = Rotation.from_matrix(ambf_pose_m_lps[:3, :3]).as_euler("xyz")

        # The volume will be parented to the anatomical_origin and offset by the origin_m value        
        vol_pos_x, vol_pos_y, vol_pos_z = origin_m * scale
        vol_rot_r,vol_rot_p,vol_rot_y = Rotation.from_matrix(np.eye(3)).as_euler("xyz")

        # doing this manually here as it is short and the yaml module is not supported/installed by default in slicer's python
        lines = []
        lines.append(f"# AMBF Version: (0.1)")
        lines.append(f"bodies:")
        lines.append(f"- BODY {volume_name}_anatomical_origin")
        lines.append(f"joints: []")
        lines.append(f"volumes: [VOLUME {volume_name}]")
        lines.append(f"high resolution path: meshes/high_res/")
        lines.append(f"low resolution path: meshes/low_res/")
        lines.append(f"ignore inter-collision: true")
        lines.append(f"namespace: /ambf/env/")
        lines.append(f"VOLUME {volume_name}:")
        lines.append(f"  name: {volume_name}")
        lines.append(f"  parent: {volume_name}_anatomical_origin")
        lines.append(f"  location:")
        lines.append(f"    position: {{x: {vol_pos_x} , y: {vol_pos_y}, z: {vol_pos_z}}}")
        lines.append(f"    orientation: {{r: {vol_rot_r}, p: {vol_rot_p}, y: {vol_rot_y}}}")
        lines.append(f"  scale: {scale}")
        lines.append(f"  dimensions: {{x: {dimensions[0]}, y: {dimensions[1]}, z: {dimensions[2]}}}")
        lines.append(f"  images:")
        lines.append(f"    path: {volume_name}/")
        lines.append(f"    prefix: {prefix}")
        lines.append(f"    format: png")
        lines.append(f"    count: {data_size[2]}")  # Number of images / slices
        lines.append(f"BODY {volume_name}_anatomical_origin: # This is a dummy body that can be used to represent the anatomical origin for easy reference") 
        lines.append(f"  name: {volume_name}_anatomical_origin")
        lines.append(f"  mass: 0.0")
        lines.append(f"  location:")
        lines.append(f"      position:")
        lines.append(f"        x: {anor_pos_x}")
        lines.append(f"        y: {anor_pos_y}")
        lines.append(f"        z: {anor_pos_z}")
        lines.append(f"      orientation:")
        lines.append(f"        r: {anor_rot_r}")
        lines.append(f"        p: {anor_rot_p}")
        lines.append(f"        y: {anor_rot_y}")


        yaml_name = os.path.join(yaml_save_location, volume_name+".yaml")
        with open(yaml_name, 'w') as f:
            f.write('\n'.join(lines))
            f.close()
        print("Saved YAML file to: " + yaml_name)

    # TODO: leaving this here for now, but it is probably better to just save natively in Slicer and 
    # then parse that file in AMBF, since they now have a json file output that includes lots of useful info
    # such as lps/ras, etc.
    def exportMarkupToCSV(self, markupNode, volumeNode, output_dir, output_name, ambf_scale=1.0):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        markup_points = slicer.util.arrayFromMarkupsControlPoints(markupNode)

        # convert to LPS by negating the x and y coordinates
        markup_points_lps = np.copy(markup_points)
        markup_points_lps[:,0] = -markup_points_lps[:,0]
        markup_points_lps[:,1] = -markup_points_lps[:,1]

        # convert to SI units
        markup_points_lps_m = markup_points_lps * 0.001

        # convert to AMBF units
        markup_points_lps_m_ambf = markup_points_lps_m * ambf_scale

        # save to csv file
        # check if outputname has .csv extension, add it if not
        if not output_name.endswith(".csv"):
            output_name = output_name + ".csv"
        csv_name = os.path.join(output_dir, output_name)
        np.savetxt(csv_name, markup_points_lps_m_ambf, delimiter=",")
        print("Saved CSV file to: " + csv_name)
    

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
        # TODO: add tests here
        self.delayDisplay("No tests for now!")
        self.delayDisplay('Test passed!')
