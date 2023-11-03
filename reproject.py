import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.plot import show
from PIL import Image, ImageOps


def reproj_match(infile, match, outfile=None, keep_shape=False):
    """Reproject a file to match the shape and projection of existing raster
    mostly from https://pygis.io/docs/e_raster_resample.html
    
            Parameters:
                    infile (string): path to input file to reproject
                    match (string): path to raster with desired shape and projection 
                    outfile (string or None): path to output file tif (if set to None,
                            the resulting raster will be returned as np array)
                    keep_shape (bool): if True, the resulting raster will have the exact
                            same dimensions as the match-raster

            Returns:
                    if outfile is None:
                        result (np.array): reprojected raster as array
    """
    # open input
    with rasterio.open(infile) as src:
        
        # open input to match
        with rasterio.open(match) as match_ds:

            dst_crs = match_ds.crs
            dst_nodata = match_ds.nodata

            kwargs = {}
            if keep_shape:
                kwargs = {"dst_width": match_ds.width,
                          "dst_height": match_ds.height}
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match_ds.width,   # input width
                match_ds.height,  # input height 
                *match_ds.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                **kwargs,
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": dst_nodata})
        
        result = np.zeros((src.count, dst_height, dst_width))
        _, _ = reproject(
            source=src.read(),
            destination=result,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=-8888,
            dst_nodata=-8888,
        )

        if outfile is not None:
            with rasterio.open(outfile, "w", **dst_kwargs) as dst:
                dst.write(result)
            return
        return result


def slice_img(input_file, size, name, out_folder):
    with rasterio.open(input_file) as src:
        im = src.read()
        im[im < 0] = 0
        im = im.squeeze()

        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        i = 0
        for tile in tiles:
            Image.fromarray(tile).save(os.path.join(out_folder, f"{i}_{name}.tiff"))
            i += 1


def remove_non_pairs(folder_path, name1, name2):
    tile1_ids = [f.split("_")[0] for f in os.listdir(folder_path) if f.endswith(f"{name1}.tiff")]
    for tile2_id in [f.split("_")[0] for f in os.listdir(folder_path) if f.endswith(f"{name2}.tiff")]:
        if tile2_id not in tile1_ids:
            os.remove(os.path.join(folder_path, f"{tile2_id}_{name2}.tiff"))

    tile2_ids = [f.split("_")[0] for f in os.listdir(folder_path) if f.endswith(f"{name2}.tiff")]
    for tile1_id in [f.split("_")[0] for f in os.listdir(folder_path) if f.endswith(f"{name1}.tiff")]:
        if tile1_id not in tile2_ids:
            os.remove(os.path.join(folder_path, f"{tile1_id}_{name1}.tiff"))


if __name__ == "__main__":
    # """
    reproj_match(r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\vie_034_nDSM_est_filtered_new.tif",
                 r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\vie_159_nDSM_est_filtered_new.tif",
                 r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\vie_034_matched_159_new.tif")
    """
    slice_img(r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\fra_339_matched_104.tif", 256, "fra339",
              r"D:\Work\tum\ChangeDet\data\fra339")
    slice_img(r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\fra_104_nDSM_est_filtered.tif", 256, "fra104",
              r"D:\Work\tum\ChangeDet\data\fra104")
    # """
    # remove_non_pairs(r"D:\Work\tum\ChangeDet\data\fra339_fra104", "fra339", "fra104")

