import os
import io
import tarfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio.plot
import yaml

# -----------------------------------------------------------------------------
#                               showfig
# -----------------------------------------------------------------------------
def showfig(data,flag = None,figsize=(5,5),ax=None):
    colorbar = ax is None
    ax = ax or plt.figure(figsize=figsize)
    if flag:
        img = (data & flag) == flag
    else:
        img = data
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
        
# -----------------------------------------------------------------------------
#                         show_single_result
# -----------------------------------------------------------------------------
def show_single_result(image_data, is_ndvi=False):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    EXAMPLE             
    res=connection.load_collection(s2.s2_msi_l2a,
                         spatial_extent=s2.bbox.karlstad_mini_land,
                         temporal_extent=s2.timespans.one_image,
                        bands=['b08','b04']
                        )
    image_data = res.download(format="gtiff")
    show_single_result(image_data)
    
    '''
    filelike = io.BytesIO(image_data)
    im = rasterio.open(filelike)
    
    fig, ax = plt.subplots()
    if is_ndvi:
        rasterio.plot.show(im, ax=ax, cmap='RdYlGn',vmin=-0.8,vmax=0.8)
    else:
        rasterio.plot.show(im, ax=ax, cmap="pink")
    ax.set_title(im.tags()["datetime_from_dim"])
    plt.title(im.tags()["datetime_from_dim"])
    return [im]
# -----------------------------------------------------------------------------
#                           show_zipped_results
# -----------------------------------------------------------------------------
def show_zipped_results(image_data, is_ndvi=False):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    EXAMPLE             
    res=connection.load_collection(s2.s2_msi_l2a,
                         spatial_extent=s2.bbox.karlstad_mini_land,
                         temporal_extent=s2.timespans.five_images,
                        bands=['b08','b04']
                        )
    image_data = res.download(format="gtiff")
    show_single_result(image_data)
    
    
    '''
    images = []
    file_like_object = io.BytesIO(image_data)
    
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Step 2: Open the tar.gz file
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            # Step 3: Extract all the contents into a specific directory
            tar.extractall(tmpdirname)
            
            if os.path.isdir(tmpdirname):
                tmpdirname = f"{tmpdirname}/{os.listdir(tmpdirname)[0]}"
                
            
            
            image_types = [".tif"]

            ifnames = [ifname for ifname in sorted(os.listdir(tmpdirname))
                       if  any(image_type in ifname for image_type in image_types)]
            
            columns = 2  # For example, adjust based on your preference and screen size

            # Calculate the required number of rows to fit all images
            rows = len(ifnames) // columns + (len(ifnames) % columns > 0)

            # Create subplots
            fig, axs = plt.subplots(rows, columns)

                                        
            for idx, ifname in enumerate(ifnames):
                fname = f"{tmpdirname}/{ifname}"
                src = rasterio.open(fname)
                images.append(src)
                if rows > 1:
                    ax = axs[idx // columns][idx % columns]
                else:
                    ax = axs[idx]

                if is_ndvi:
                    rasterio.plot.show(src, ax=ax, cmap='RdYlGn',vmin=-0.8,vmax=0.8)
                else:
                    cmap = plt.cm.inferno  
                    cmap.set_bad('black', 1.0) 
                    rres = rasterio.plot.show(src.read(1), transform=src.transform, ax=ax, cmap=cmap, vmin=0, vmax = 2)
                    im = rres.get_images()[0]
                    fig.colorbar(im, ax=ax, shrink=0.35, aspect=10)
                ax.set_title(src.tags()["datetime_from_dim"])
        return images
# -----------------------------------------------------------------------------
#                               show_result
# -----------------------------------------------------------------------------
def show_result(image_data, is_ndvi=False):
    try:
        return show_single_result(image_data, is_ndvi)
    except Exception as e:
        pass
    return show_zipped_results(image_data, is_ndvi)

# -----------------------------------------------------------------------------
#                               get_s3_wqsf_flags
# -----------------------------------------------------------------------------
def get_s3_wqsf_flags():
    '''
    You can get these flags from get_collections, but this is a shortcut for
    training purposes.
    
    '''
    wqsf_flags = {}
    here = Path(__file__).parent
    with open(f"{here}/s3_olci_l2wfr.odc-product.yaml", 'r') as stream:
        s3_meta = yaml.safe_load(stream)
      
        for m in s3_meta['measurements']:
            if 'wqsf' in m['name']:
                bits = (m['flags_definition']['data']['values'])
                bitmap = {}
                for b in bits.keys():
                    bitmap[bits[b]] = b 
                    
                wqsf_flags[m['name']] = bitmap
    return wqsf_flags
