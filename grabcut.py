
# coding: utf-8

# In[ ]:


import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import holoviews as hv
from holoviews import opts
from holoviews import streams
hv.extension('bokeh')


# In[ ]:


img_folder = '/Volumes/passport_hd/misc_data/sg/staff_2018/individual'
img = cv.imread(os.path.join(img_folder, 'darren.jpg'))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# In[ ]:


def plot(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.show()


# In[ ]:


rgb = hv.RGB(img, bounds=(0, 0, img.shape[1], img.shape[0]))

aoi_box = hv.Polygons([])
aoi_stream = streams.BoxEdit(source=aoi_box, num_objects=1)
aoi_box.opts(opts.Polygons(active_tools=['box_edit'], fill_alpha=0.5, height=300, width=260))

keep_path = hv.Path([]).opts(color='green', line_width=5)
keep_freehand = streams.FreehandDraw(source=keep_path)
keep_path.options(opts.Path(active_tools=['freehand_draw']))

discard_path = hv.Path([]).opts(color='red', line_width=5)
discard_freehand = streams.FreehandDraw(source=discard_path)
discard_path.options(opts.Path(active_tools=['freehand_draw']))

hv.Overlay([rgb, aoi_box, keep_path, discard_path])


# In[ ]:


mask = np.zeros(img.shape[:2],np.uint8) +2

keep_data = keep_freehand.data
keep_x = [int(item) for sublist in keep_data['xs'] for item in sublist]
keep_y = [int(item) for sublist in keep_data['ys'] for item in sublist]
keep_points = np.array(list(zip(keep_x, keep_y)))

for i in keep_points:
    mask[img.shape[0] - i[1], i[0]] = 1
    
discard_data = discard_freehand.data
discard_x = [int(item) for sublist in discard_data['xs'] for item in sublist]
discard_y = [int(item) for sublist in discard_data['ys'] for item in sublist]
discard_points = np.array(list(zip(discard_x, discard_y)))

for i in discard_points:
    mask[img.shape[0] - i[1], i[0]] = 0
    
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

aoi_stream_data = aoi_stream.data

cv.grabCut(rgb.data,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
mask2 = cv.dilate(mask2, None, iterations=10)
mask2 = cv.erode(mask2, None, iterations=10)
img_copy = img*mask2[:,:,np.newaxis]

rect = (
    int(aoi_stream_data['x0'][0]),
    int(img.shape[0] - aoi_stream_data['y1'][0]),
    int(aoi_stream_data['x1'][0]),
    int(img.shape[0] - aoi_stream_data['y0'][0])
)

img_copy = img_copy[rect[1]:rect[3], rect[0]:rect[2], :]
print(img_copy.shape)
plot(img_copy)
hv.RGB(img_copy)

