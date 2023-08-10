# tree_detection
An attempt to utilize a tree detection model for future projects

This is a Detectron2 based tree detection model with training data retrieved from Zenodo for tree count and representation purposes. 
The following packages need to be used:
1) cv2
2) PIL
3) os
4) numpy
5) urllib.request
6) glob
7) intake
8) intake_zenodo_fetcher.intake_zenodo_fetcher import download_zenodo_files_for_entry
9) geopandas
10) rasterio
11) fiona
12) shapely
13) detectron2
14) holoviews
15) geoviews
16) hvplot

The model that was used requires the following steps for setup:
1. Create a new environment in Python 3.11
2. Download Detectron2 (https://github.com/facebookresearch/detectron2)
3. Download Intake Zenodo Fetcher (https://github.com/ESM-VFC/intake_zenodo_fetcher)
4. Download Detectree Model (https://github.com/shmh40/detectreeRGB/tree/main)
5. Make sure that everything is held up in a single folder
6. Change data_folder path to the folder with the data that you have
7. Make sure to have the following folders inside:
  1) input
  2) output
  3) model
8. Make sure to put the .tiff in the "tiff" folder and save a copy of a .tiff as a .png to put it in the "png" folder (temporary measure since the code is not able to render the tiff later)
9. Change the path to the .tiff in "urlpath" (line 69)
10. Change lines 86-97 according to the settings of your image
11. Change the "filepath" (line 110) to your png
12. Run the code

As a result, you should have 2 usable files:
1) .html file that shows you the rendered image in your browser
2) A shapefile in the following path "./forest-modelling-detectree/output/vector/"


Please, note that the generated .html currently shows a projection error and the shapefile on top seems to be shifted. The shapefile itself looks good when used with a .tiff in any GIS tool.
