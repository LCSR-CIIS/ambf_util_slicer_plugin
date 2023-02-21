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
        slicer.mrmlScene.AddNode(self.AMBF_X)
        self.AMBF_Y = slicer.vtkMRMLMarkupsLineNode()
        self.AMBF_Y.SetName("AMBF_Y")
        slicer.mrmlScene.AddNode(self.AMBF_Y)
        self.AMBF_Z = slicer.vtkMRMLMarkupsLineNode()
        self.AMBF_Z.SetName("AMBF_Z")
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

        # volumeName is disabled if the checkbox is not checked
        self.volumeName.enabled = self.generateYaml.checked
        self.ambfScale.enabled = self.generateYaml.checked
        # connect the checkbox to the volumeName to enable/disable it
        self.generateYaml.connect('stateChanged(int)', self.onGenerateYamlCheckbox)
        # disable the volumeName if the checkbox is not checked

        # Create a button to export the LabelMap Node to png slices
        self.exportLabelMapButton = qt.QPushButton("Export LabelMap to PNGs for AMBF")
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

        # create a transform node called "anatomical_T_AMBF"
        self.anatomical_T_AMBF = slicer.vtkMRMLLinearTransformNode()
        self.anatomical_T_AMBF.SetName("Anatomical_to_AMBF_Origin")
        slicer.mrmlScene.AddNode(self.anatomical_T_AMBF)
        # make a transform node called "AMBF_T_anatomical"
        self.AMBF_T_anatomical = slicer.vtkMRMLLinearTransformNode()
        self.AMBF_T_anatomical.SetName("AMBF_Origin_to_Anatomical")
        slicer.mrmlScene.AddNode(self.AMBF_T_anatomical)
        # set this node to always contain the transform from the current volume origin to its center point
        self.referenceVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onReferenceVolumeChanged)

        self.ambfPose = slicer.vtkMRMLLinearTransformNode()
        self.ambfPose.SetName("AMBF_Pose")
        slicer.mrmlScene.AddNode(self.ambfPose)

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
            self.generateYaml.checked, self.volumeName.text, self.ambfScale.value, self.ambfPose)

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
    
    def onSegmentLabelMapChanged(self):
        # set anatomical_T_AMBF to the transform from the current volume origin to its center point
        volumeNode = self.segmentLabelMapSelector.currentNode()
        if volumeNode is None:
            return

        # Here, I will use the notation A_T_B, A_R_B, A_p_B, etc. This is a shorthand for transform, rotation, and position that take
        # points in frame B and express them in frame A. This is convenient because A_T_C = A_T_B * B_T_C in this notation.
        AMBFOriginInSlicer_p_spaceOriginInSlicer = self.logic.calculate_AMBFOriginInSlicer_p_spaceOriginInSlicer(volumeNode)
        spaceOriginInSlicer_T_AMBFOriginInSlicer = np.eye(4)
        spaceOriginInSlicer_T_AMBFOriginInSlicer[0:3,3] = -AMBFOriginInSlicer_p_spaceOriginInSlicer
        slicer.util.updateTransformMatrixFromArray(self.anatomical_T_AMBF, spaceOriginInSlicer_T_AMBFOriginInSlicer)
        slicer.util.updateTransformMatrixFromArray(self.AMBF_T_anatomical, np.linalg.inv(spaceOriginInSlicer_T_AMBFOriginInSlicer))

        self.update_AMBF_axes(AMBFOriginInSlicer_p_spaceOriginInSlicer)
        self.updateTransformRelations()

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
        # volume and labelmap nodes should be relative to anatomical_T_ambf
        # anatomical_T_ambf should be relative to ambf_pose
        # ambf_pose should be relative to AMBF_T_anatomical

        # get the current nodes
        volumeNode = self.referenceVolumeSelector.currentNode()
        labelMapNode = self.segmentLabelMapSelector.currentNode()

        # if the volume node is not None, set its transform to anatomical_T_AMBF
        if volumeNode is not None:
            volumeNode.SetAndObserveTransformNodeID(self.anatomical_T_AMBF.GetID())
        # if the labelmap node is not None, set its transform to anatomical_T_AMBF
        if labelMapNode is not None:
            labelMapNode.SetAndObserveTransformNodeID(self.anatomical_T_AMBF.GetID())
        # if the ambf pose node is not None, set its transform to AMBF_T_anatomical
        self.ambfPose.SetAndObserveTransformNodeID(self.AMBF_T_anatomical.GetID())
        self.anatomical_T_AMBF.SetAndObserveTransformNodeID(self.ambfPose.GetID())


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

    def calculate_AMBFOriginInSlicer_p_spaceOriginInSlicer(self, volumeNode):
        # assumes AMBF is at volume center with x,y,z axes aligned with LPS
        # get volume center in RAS coordinates (some help from https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html)
        volumeArray = slicer.util.arrayFromVolume(volumeNode) # NOTE: this utility makes an array in K,J,I order, not I,J,K
        center_kji = np.array(volumeArray.shape)/2
        center_ijk = np.flip(center_kji)
        IJKToRASMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(IJKToRASMatrix)
        volumeCenter_p_spaceOrigin = np.array(IJKToRASMatrix.MultiplyPoint(np.append(center_ijk,1.0))[0:3])
        return volumeCenter_p_spaceOrigin


    def exportLabelMapToPNG(self, labelMapNode, outputDir, image_prefix, grayscale, generateYaml, volume_name, scale, ambf_pose_node):
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
        
        # THIS IS VOLUME ORIGIN
        origin_mm = (origin - (dimensions_mm/2))
        origin_m = 0.001 * origin_mm

        # THIS IS ANATOMICAL ORIGIN
        AMBFOriginInSlicer_p_spaceOriginInSlicer = self.calculate_AMBFOriginInSlicer_p_spaceOriginInSlicer(labelMapNode)
        print("AMBFOriginInSlicer_p_spaceOriginInSlicer: " + str(AMBFOriginInSlicer_p_spaceOriginInSlicer))
        anatomical_origin_m = -AMBFOriginInSlicer_p_spaceOriginInSlicer * 0.001
    
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
            self.save_yaml_file(data_size, dimensions_m, volume_name, yaml_save_location, anatomical_origin_m, scale, image_prefix, ambf_pose_node)


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
        data_size = np.array(data_size)
        # unpack ambf pose to numpy array
        ambf_pose = slicer.util.arrayFromTransformMatrix(ambf_pose_node)
        # convert to m
        ambf_pose_m = ambf_pose
        ambf_pose_m[0:3,3] = ambf_pose_m[0:3,3] * 0.001
        # convert to lps convention
        ras2lps = np.diag([-1, -1, 1, 1])
        ambf_pose_m_lps = ras2lps @ ambf_pose_m @ ras2lps
        # convert to m, then scale

        # we want to offset the volume origin by this amount and also adjust the anatomical_origin body respectively
        # internally AMBF uses a R,P,Y convention which equates to an Euler angle Z,Y,X where R is x P is y, Y is z
        #determine offset using homogeneous transformation ambf_pose @ [I origin; 0 0 0 1]
        old_origin_m = np.eye(4)
        old_origin_m[0:3,3] = origin_m
        # convert to lps convention
        old_origin_m_lps = ras2lps @ old_origin_m @ ras2lps
        new_origin_m_lps = ambf_pose_m_lps @ old_origin_m_lps

        from scipy.spatial.transform import Rotation

        vol_pos_x, vol_pos_y, vol_pos_z = ambf_pose_m_lps[0:3,3] * scale
        vol_rot_r,vol_rot_p,vol_rot_y = Rotation.from_matrix(ambf_pose_m_lps[:3, :3]).as_euler("xyz")

        origin_pos_x, origin_pos_y, origin_pos_z = new_origin_m_lps[0:3,3] * scale
        origin_rot_r, origin_rot_p, origin_rot_y = Rotation.from_matrix(new_origin_m_lps[:3, :3]).as_euler("xyz")

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
        lines.append(f"  location:")
        lines.append(f"    position: {{x: {vol_pos_x} , y: {vol_pos_y}, z: {vol_pos_z}}}")
        lines.append(f"    orientation: {{r: {vol_rot_r}, p: {vol_rot_p}, y: {vol_rot_y}}}")
        lines.append(f"  scale: {scale}")
        lines.append(f"  dimensions: {{x: {dimensions[0]}, y: {dimensions[1]}, z: {dimensions[2]}}}")
        lines.append(f"  images:")
        lines.append(f"    path: {volume_name}/")
        lines.append(f"    prefix: {prefix}")
        lines.append(f"    format: png")
        lines.append(f"    count: {np.max(data_size)}")  # Note this can be larger than actual value
        lines.append(f"BODY {volume_name}_anatomical_origin: # This is a dummy body that can be used to represent the anatomical origin for easy reference") 
        lines.append(f"  name: {volume_name}_anatomical_origin")
        lines.append(f"  mass: 0.0")
        lines.append(f"  location:")
        lines.append(f"      position:")
        lines.append(f"        x: {origin_pos_x}")
        lines.append(f"        y: {origin_pos_y}")
        lines.append(f"        z: {origin_pos_z}")
        lines.append(f"      orientation:")
        lines.append(f"        r: {origin_rot_r}")
        lines.append(f"        p: {origin_rot_p}")
        lines.append(f"        y: {origin_rot_y}")


        yaml_name = os.path.join(yaml_save_location, volume_name+".yaml")
        with open(yaml_name, 'w') as f:
            f.write('\n'.join(lines))
            f.close()
        print("Saved YAML file to: " + yaml_name)

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
        self.delayDisplay("No tests for now!")
        self.delayDisplay('Test passed!')
