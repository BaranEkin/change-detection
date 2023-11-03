import rasterio
from PIL import Image
import numpy as np
from rasterio.warp import reproject, Resampling, calculate_default_transform
from torchvision import transforms
from model import PatchChangeDet128


def calculate_change(model, size, img1_np, img2_np):
    """Calculates change score in between two height map patches.

    Height maps should be numpy arrays of model's input shape.
    """

    img1 = Image.fromarray(img1_np)
    img2 = Image.fromarray(img2_np)

    transform = transforms.Compose([transforms.Resize((size, size)), transforms.PILToTensor()])
    img1 = transform(img1).unsqueeze(dim=0).to("cuda")
    img2 = transform(img2).unsqueeze(dim=0).to("cuda")

    model.eval()
    return model(img1, img2).item()


def create_change_image(model, size, stride, img1_path, img2_path, output_path, change_threshold=0.3,
                        interpolation=Image.BILINEAR):
    """Creates a change map given two input height maps.

    Height maps first should be reprojected to match.
    """

    with rasterio.open(img1_path) as img1_file:
        with rasterio.open(img2_path) as img2_file:

            img1 = img1_file.read()
            img2 = img2_file.read()

            img1 = img1.squeeze()
            img2 = img2.squeeze()

            change_scores_np = np.zeros(((img1.shape[0] - size) // stride + 1, (img1.shape[1] - size) // stride + 1))

            for y in range(0, img1.shape[1] - size, stride):
                print(f"\rComputing... {y / img1.shape[1] * 100:.1f}%", end="")
                for x in range(0, img1.shape[0] - size, stride):
                    tile_img1_np = img1[x:x + size, y:y + size]
                    tile_img2_np = img2[x:x + size, y:y + size]

                    if (tile_img1_np == -8888).sum() > 0 or (tile_img2_np == -8888).sum() > 0:
                        change_scores_np[x // stride, y // stride] = 0  # -1

                    else:
                        tile_img1_np[tile_img1_np < 0] = 0
                        tile_img2_np[tile_img2_np < 0] = 0

                        change_score = calculate_change(model, size, tile_img1_np, tile_img2_np)
                        # change_score = mask_sniper(tile_img1_np, tile_img2_np, 64, 8)
                        change_scores_np[x // stride, y // stride] = change_score

            change_scores_np = change_scores_np - change_threshold
            change_scores_np[change_scores_np < 0] = 0

            print(f"\rComputing... 100%", end="")
            change_img = Image.fromarray(change_scores_np).resize((img1.shape[1], img1.shape[0]), interpolation)
            change_img_np = np.asarray(change_img)

            dst_crs = img1_file.crs
            dst_nodata = img1_file.nodata

            kwargs = {}
            kwargs = {"dst_width": img1_file.width,
                      "dst_height": img1_file.height}

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                img1_file.crs,  # input CRS
                dst_crs,  # output CRS
                img1_file.width,  # input width
                img1_file.height,  # input height
                *img1_file.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                **kwargs,
            )

            # set properties for output
        dst_kwargs = img1_file.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": dst_nodata})

        result = np.zeros((img1_file.count, dst_height, dst_width))
        _, _ = reproject(
            source=change_img_np,
            destination=result,
            src_transform=img1_file.transform,
            src_crs=img1_file.crs,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=img1_file.nodata,
            dst_nodata=dst_nodata,
        )

        with rasterio.open(output_path, "w", **dst_kwargs) as dst:
            dst.write(result)


checkpoint_path = r"PatchChangeDet128_v4-epoch=40-val_loss=0.54.ckpt"
model = PatchChangeDet128.load_from_checkpoint(checkpoint_path, train_set=None, val_set=None, batch_size=1)

"""

create_change_image(model, 128, 16,
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\fra_339_matched_903_new.tif",
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\fra_903_nDSM_est_filtered_new.tif",
                    r"D:\Work\tum\ChangeDet\results\final_georeferenced\fra_339_903.tiff",
                    change_threshold=0)

create_change_image(model, 128, 16,
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\ber_452_matched_612_new.tif",
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\ber_612_nDSM_est_filtered_new.tif",
                    r"D:\Work\tum\ChangeDet\results\final_georeferenced\ber_452_612.tiff",
                    change_threshold=0)

create_change_image(model, 128, 16,
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\muc_121_matched_714_new.tif",
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\muc_714_nDSM_est_filtered_new.tif",
                    r"D:\Work\tum\ChangeDet\results\final_georeferenced\muc_121_714.tiff",
                    change_threshold=0)

create_change_image(model, 128, 16,
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\muc_121_matched_749_new.tif",
                    r"D:\Work\tum\ChangeDet\data\changedet_generated_nDSMs\muc_749_nDSM_est_filtered_new.tif",
                    r"D:\Work\tum\ChangeDet\results\final_georeferenced\muc_121_749.tiff",
                    change_threshold=0)

"""