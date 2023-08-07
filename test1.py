import cv2
from PIL import Image
import os
import numpy as np
import urllib.request
import glob

# intake library and plugin
import intake
from intake_zenodo_fetcher.intake_zenodo_fetcher import download_zenodo_files_for_entry

# geospatial libraries
import geopandas as gpd

from rasterio.transform import from_origin
import rasterio.features

import fiona

from shapely.geometry import shape, mapping, box
from shapely.geometry.multipolygon import MultiPolygon

# machine learning libraries
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

# visualisation
import holoviews as hv
from IPython.display import display
import geoviews.tile_sources as gts

import hvplot.pandas
import hvplot.xarray

hv.extension('bokeh', width=100)


# Define the project main folder
data_folder = './forest-modelling-detectree'

# Set the folder structure
config = {
    'in_geotiff': os.path.join(data_folder, 'input','tiff'),
    'in_png': os.path.join(data_folder, 'input','png'),
    'model': os.path.join(data_folder, 'model'),
    'out_geotiff': os.path.join(data_folder, 'output','raster'),
    'out_shapefile': os.path.join(data_folder, 'output','vector'),
}

# List comprehension for the folder structure code
[os.makedirs(val) for key, val in config.items() if not os.path.exists(val)]


# write a catalog YAML file
catalog_file = os.path.join(data_folder, 'catalog.yaml')

with open(catalog_file, 'w') as f:
    f.write('''
sources:
  sepilok_rgb:
    driver: rasterio
    description: 'NERC RGB images of Sepilok, Sabah, Malaysia (collection)'
    metadata:
      zenodo_doi: "10.5281/zenodo.5494629"
    args:
      urlpath: "{{ CATALOG_DIR }}/input/tiff/Sep_2014_RGB_602500_646600.tif"
      ''')

cat_tc = intake.open_catalog(catalog_file)

for catalog_entry in list(cat_tc):
    download_zenodo_files_for_entry(
        cat_tc[catalog_entry],
        force_download=False
    )

tc_rgb = cat_tc["sepilok_rgb"].to_dask()

print('shape =', tc_rgb.shape,',', 'and number of bands =', tc_rgb.count, ', crs =', tc_rgb.crs)

# Set minx and miny depending on the tif coordinates
#411436.5959040710586123,3903562.5368613298051059 : 411976.5119040710851550,3904102.4528613300062716
minx = 411844
miny = 3903987

R = tc_rgb[0]
G = tc_rgb[1]
B = tc_rgb[2]
    
# stack up the bands in an order appropriate for saving with cv2, then rescale to the correct 0-255 range for cv2

# you will have to change the rescaling depending on the values of your tiff!
rgb = np.dstack((R,G,B)) # BGR for cv2
rgb_rescaled = 255*rgb/(100000*100000) # scale to image
    
# # save this as png, named with the origin of the specific tile - change the filepath!
# filepath = config['in_png'] + '/' + 'tile_' + str(minx) + '_' + str(miny) + '.png'
# cv2.imwrite(filepath, rgb_rescaled)

# # Trying to set a separate dataset for the tiled tif, failed. Does not transform to the required crs
# in_tifpath = 'C:/Users/First/Documents/AeroTract/DT2_test/forest-modelling-detectree/input/tiff/Sep_2014_RGB_602500_646600.tif'
# dataset1 = gdal.Open(in_tifpath)
# print(dataset1)
# projection = dataset1.GetProjection()
# geotransform = dataset1.GetGeoTransform()

filepath = 'C:/Users/First/Documents/AeroTract/DT2_test/forest-modelling-detectree/input/png/tile_411437_3903563.png' #temporary setting measure, since tiling doesn't work
im = cv2.imread(filepath)
display(Image.fromarray(im))

# define the URL to retrieve the model
fn = 'model_final.pth'
url = f'https://zenodo.org/record/5515408/files/{fn}?download=1'

urllib.request.urlretrieve(url, config['model'] + '/' + fn)


cfg = get_cfg()

# if you want to make predictions using CPU, run the following line. If using CUDA, hash it out.
#cfg.MODEL.DEVICE='cpu'

# model and hyperparameter selection
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

### path to the saved pre-trained model weights
cfg.MODEL.WEIGHTS = config['model'] + '/model_final.pth'

# set confidence threshold at which we predict
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

#### Settings for predictions using detectron config

predictor = DefaultPredictor(cfg)


outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], scale=1.5, instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
image = cv2.cvtColor(v.get_image()[:, :, :], cv2.COLOR_BGR2RGB)
display(Image.fromarray(image))


mask_array = outputs['instances'].pred_masks.cpu().numpy()

# Confidence scores
mask_array_scores = outputs['instances'].scores.cpu().numpy()

num_instances = mask_array.shape[0]
mask_array_instance = []
output = np.zeros_like(mask_array) 

mask_array_instance.append(mask_array)
output = np.where(mask_array_instance[0] == True, 255, output)
fresh_output = output.astype(np.float)
x_scaling = 140/fresh_output.shape[1]
y_scaling = 140/fresh_output.shape[2]

# Transform requires manual modifications:
# Filepath: change the values inside the brackets according to minx and miny
# Change the number after depending on the padding you wish to include
transform = from_origin(int(filepath[-18:-12])+100, int(filepath[-11:-4])+100, y_scaling, x_scaling)

output_raster = config['out_geotiff'] + '/' + 'predicted_rasters_' + filepath[-18:-4]+ '.tif'

new_dataset = rasterio.open(output_raster, 'w', driver='GTiff',
                                height = fresh_output.shape[1], width = fresh_output.shape[2], count = fresh_output.shape[0],
                                dtype=str(fresh_output.dtype),
                                crs='+proj=utm +zone=12 +datum=NAD83 +units=m +no_defs=True',  
                                transform=transform)

new_dataset.write(fresh_output)
new_dataset.close()


# Read input band with Rasterio
    
with rasterio.open(output_raster) as src:
    shp_schema = {'geometry': 'MultiPolygon','properties': {'pixelvalue': 'int', 'score': 'float'}}    

    crs = src.crs
    for i in range(src.count):
        src_band = src.read(i+1)
        src_band = np.float32(src_band)
        conf = mask_array_scores[i]
        # Keep track of unique pixel values in the input band
        unique_values = np.unique(src_band)
        # Polygonize with Rasterio. `shapes()` returns an iterable
        # of (geom, value) as tuples
        shapes = list(rasterio.features.shapes(src_band, transform=src.transform))

        if i == 0:
            with fiona.open(config['out_shapefile'] + '/predicted_polygons_' + filepath[-18:-4] + '_' + str(0) + '.shp', 'w', 'ESRI Shapefile',
                            shp_schema) as shp:
                polygons = [shape(geom) for geom, value in shapes if value == 255.0]                                        
                multipolygon = MultiPolygon(polygons)                  
                shp.write({
                          'geometry': mapping(multipolygon),
                          'properties': {'pixelvalue': int(unique_values[1]), 'score': float(conf)} 
                           })
        else:
            with fiona.open(config['out_shapefile'] + '/predicted_polygons_' + filepath[-18:-4] + '_' + str(0)+'.shp', 'a', 'ESRI Shapefile',
                            shp_schema) as shp:
                polygons = [shape(geom) for geom, value in shapes if value == 255.0]                                        
                multipolygon = MultiPolygon(polygons)                  
                shp.write({
                          'geometry': mapping(multipolygon),
                          'properties': {'pixelvalue': int(unique_values[1]), 'score': float(conf)} 
                           })



# load and plot polygons
in_shp = glob.glob(config['out_shapefile'] + '/*.shp')

poly_df = gpd.read_file(in_shp[0])

plot_vector = poly_df.hvplot(hover_cols=['score'], legend=False).opts(fill_color=None,line_color=None,alpha=0.5, width=800, height=600)

plot_vector


# load and plot RGB image
r = tc_rgb.sel(band=[1,2,3])

normalized = r/(r.quantile(.99,skipna=True)/255)

mask = normalized.where(normalized < 255)

int_arr = mask.astype(int)

plot_rgb = int_arr.astype('uint8').hvplot.rgb(
    x='x', y='y', bands='band', data_aspect=0.8, hover=False, legend=False, rasterize=True
)


plot_rgb * plot_vector


combined_plot = plot_rgb * plot_vector

hvplot.save(combined_plot, data_folder+'/combined_plot.html')
