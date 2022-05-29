import numpy as np
import glob 
import maxflow
import torch
import matplotlib.pyplot as plt
from model import UNET
from sklearn.cluster import KMeans
from albumentations.pytorch import ToTensorV2
import albumentations as A
from skimage.io import imsave
from dataset import SkinDataset
from torch.utils.data import DataLoader 
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")
from utils import (
    _get_global_confusion_items,  
    _compute_segment_metrics,  
    ConfusionItems, 
    compute_confusion_results,
    get_test_loaders,
    load_checkpoint
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 1
IMAGE_HEIGHT = 288 
IMAGE_WIDTH = 384
OPT = 0 # 0: filter small obj and make a dilatation of the predicted masks by Unet. 1: filter small obj, erosion & dilatations to produce forground/background regions for graph cut
PIN_MEMORY = True
PREDICTION = True
CHECKPINT_PATH = "checkpoint/my_checkpoint.pth.tar"
TEST_IMG_DIR = "data/test_images/"
# if not os.path.exists(TEST_IMG_DIR):
#     os.mkdir(TEST_IMG_DIR)
TEST_MASK_DIR = "data/test_masks/"
# if not os.path.exists(TEST_MASK_DIR):
#     os.mkdir(TEST_MASK_DIR)
FOLDER_SAVE_PREDICT="data/saved_predictions_test/"
if not os.path.exists(FOLDER_SAVE_PREDICT):
    os.mkdir(FOLDER_SAVE_PREDICT)
else:
    files = glob.glob(FOLDER_SAVE_PREDICT+'*')
    for f in files:
        os.remove(f)
UNET_MASKS_DIR = "data/predicted_Unet"
# if not os.path.exists(UNET_MASKS_DIR):
#     os.mkdir(UNET_MASKS_DIR)
FOLDER_SAVE_CROPED = "data/saved_crops_test/"
if not os.path.exists(FOLDER_SAVE_CROPED):
    os.mkdir(FOLDER_SAVE_CROPED)
else:
    files = glob.glob(FOLDER_SAVE_CROPED+'*')
    for f in files:
        os.remove(f)
mask_files = [f for f in sorted(glob.glob(UNET_MASKS_DIR+'/*.png'))]
IMAGE_NO_MASK = "data/test_images_no_mask.txt"

def get_means(img, im_bin):
    mean_object = np.array([np.mean(img[im_bin==1,0]), np.mean(img[im_bin==1,1]), np.mean(img[im_bin==1,2])])
    mean_background = np.array([np.mean(img[im_bin==0,0]), np.mean(img[im_bin==0,1]), np.mean(img[im_bin==0,2])])
    return mean_object, mean_background


def clustering(img, im_bin_ob, im_bin_bg):

    X_1 = np.vstack([img[im_bin_ob==1,i].flatten() for i in range(3)]).transpose()
    kmeans_lesion = KMeans(n_clusters=5, random_state=0).fit(X_1)
    mean_vectors_lesion = kmeans_lesion.cluster_centers_

    X_2 = np.vstack([img[im_bin_bg==0,i].flatten() for i in range(3)]).transpose()
    kmeans_background = KMeans(n_clusters=5, random_state=0).fit(X_2)
    mean_vectors_background = kmeans_background.cluster_centers_

    neg_log_likelihood_lesion_combined = np.amin(np.dstack([(sum((img[:,:,i]-mean_vectors_lesion[n_cl,i])**2 for i in range(3))) for n_cl in range(len(mean_vectors_lesion))]),2)
    neg_log_likelihood_background_combined = np.amin(np.dstack([(sum((img[:,:,i]-mean_vectors_background[n_cl,i])**2 for i in range(3))) for n_cl in range(len(mean_vectors_background))]),2)

    return neg_log_likelihood_lesion_combined, neg_log_likelihood_background_combined

def segment(im, neg_log_bg, neg_log_obj, beta = 1):

    g = maxflow.Graph[float]() # Graph instantiation
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(im.shape[0:2])
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, beta)
    # Add the terminal edges.
    g.add_grid_tedges(nodeids, neg_log_bg, neg_log_obj)

    flow = g.maxflow()

    # print("Max Flow:", str(flow))
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids) # Returns 1 if the pixel is on the drain side after calculation of the min cut, 0 if it is on the source side
    im_bin = np.int_(np.logical_not(sgm))

    return im_bin


def display_segmentation_borders(image, bin):
    imagergb = np.copy(image)
    from skimage.morphology import binary_dilation, disk
    contour = binary_dilation(bin,disk(15))^bin
    imagergb[contour==1,0] = 255
    imagergb[contour==1,1] = 0
    imagergb[contour==1,2] = 0
    return imagergb

def get_big_area(mask, x, y, empty_mask, img, img_path,opt=1):
    from skimage import morphology
    from skimage.measure import regionprops, label, regionprops_table
    import pandas as pd

    mask = np.array(Image.fromarray(mask).resize((y,x)))
    labeled = label(mask.astype(int))
    props = regionprops_table(labeled, properties=('coords','area'))
    df = pd.DataFrame(props)
    coords = list(df[df['area']==df['area'].max()]['coords'])

    if len(coords)!=0:
        labeled = labeled*0
        coords = coords[0]
        for i in range(len(coords)):
            labeled[coords[i][0],coords[i][1]]=1
        if opt==1:
            labeled_1 = morphology.dilation(labeled, morphology.disk(70))
            labeled_3 = morphology.dilation(labeled, morphology.disk(50))
            labeled_1 = labeled_1 - labeled_3
            labeled_2 = morphology.erosion(labeled, morphology.disk(5))
        else:
            labeled_1 = morphology.dilation(labeled, morphology.disk(20))
            x = list(pd.DataFrame(regionprops_table(labeled_1, properties=('bbox','bbox'))).iloc[0,:])
            labeled_2 = img[x[0]:x[2],x[1]:x[3]]  
    else:
        empty_mask.append(img_path)
        if opt==1:
            labeled_1=mask
            labeled_2=mask
        else:
            labeled_1=mask
            labeled_2=img            
    return labeled_1, labeled_2

def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),

        ],
    )

    data_loader = get_test_loaders(
        test_img_dir=TEST_IMG_DIR,
        test_mask_dir=TEST_MASK_DIR, 
        batch_size=1,
        test_transform=test_transform,
        num_workers=1
    )
    num_correct = 0
    num_pixels = 0
    IOU_score = 0
    IOU_score_wout = 0
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(CHECKPINT_PATH), model)
    model.eval()
    device='cpu'
    torch.cuda.empty_cache()
    empty_mask=[]
    nb_labels = 2
    _actual_predicted = np.zeros([nb_labels, nb_labels], dtype=np.int64)
    if PREDICTION:
        my_file = open(IMAGE_NO_MASK, "r")
        content = my_file.read()
        paths = content.split("\n")
        my_file.close()
        # Predict masks using the pretrained UNET, then postprocess results (remving small obj + delatation)
        with torch.no_grad():
            for i in range(len(paths)-1):
                img_path = paths[i]
                name = img_path.split('/')[-1].split('.')[0]
                z_ = np.array(Image.open(img_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
                z = test_transform(image=z_)['image']
                z = torch.unsqueeze(z, 0)
                im = z.to(device)
                im = im.type(torch.cuda.FloatTensor)
                im.requires_grad
                # print(type(im), im.shape, im.dtype)
                preds_ = torch.sigmoid(model(im))
                preds_1 = (preds_ > 0.5).float()
                preds_2 =preds_1.cpu().numpy().squeeze()
                if OPT == 1:
                    mask_dl, mask_er = get_big_area(preds_2, z_.shape[0], z_.shape[1], empty_mask, z_, img_path, opt=OPT)
                    neg_log_obj, neg_log_bg = clustering(z_, mask_er, mask_dl)
                    im_bin_ = segment(z_, neg_log_bg, neg_log_obj, beta=1500)
                else:
                    im_bin_, img_bbox = get_big_area(preds_2, z_.shape[0], z_.shape[1], empty_mask, z_, img_path, opt=OPT)
                print('save_prediction '+str(i)+' / '+str(len(paths)-1)+' as_imgs')
                # Ensure that the image/mask to be saved is not in the list of empty masks
                count = [name in empty_mask[j] for j in range(len(empty_mask))].count(True)
                if count == 0:
                    imsave(f"{FOLDER_SAVE_PREDICT}/{name}_seg.png", im_bin_)        
                    if OPT == 0:
                        imsave(f"{FOLDER_SAVE_CROPED}/croped_{name}.jpg", img_bbox)        

    else:
        image_files = [f for f in glob.glob(TEST_IMG_DIR+'/*.jpg')]
        with torch.no_grad():
            for i in range(len(image_files)):
                img_path = image_files[i]
                name = image_files[i].split('/')[-1].split('.')[0]
                mask_path = os.path.join(TEST_MASK_DIR, name + '_seg.png')
                z_ = np.array(Image.open(img_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
                target = np.array(Image.open(mask_path).resize((IMAGE_WIDTH,IMAGE_HEIGHT)))
                z = test_transform(image=z_)['image']
                z = torch.unsqueeze(z, 0)
                im = z.to(device)
                im = im.type(torch.FloatTensor)
                im.requires_grad
                preds_ = torch.sigmoid(model(im))
                preds_1 = (preds_ > 0.5).float()
                preds_2 =preds_1.cpu().numpy().squeeze()
                if OPT == 1:
                    mask_dl, mask_er = get_big_area(preds_2, z_.shape[0], z_.shape[1], empty_mask, z_, img_path, opt=OPT)
                    neg_log_obj, neg_log_bg = clustering(z_, mask_er, mask_dl)
                    im_bin_ = segment(z_, neg_log_bg, neg_log_obj, beta=1500)
                else:
                    im_bin_, img_bbox = get_big_area(preds_2, z_.shape[0], z_.shape[1], empty_mask, z_, img_path, opt=OPT)
                num_correct += (im_bin_ == target).sum()
                num_pixels += torch.numel(torch.tensor(im_bin_))
                IOU_score_wout += ((preds_2 * target).sum())/ ((preds_2 + target).sum() + 1e-8)            
                IOU_score += ((im_bin_ * target).sum())/ ((im_bin_ + target).sum() + 1e-8)
                print('save_prediction '+str(i)+' / '+str(len(image_files))+' as_imgs')
                # plt.figure(figsize=(10,10))
                # plt.subplot(2,2,1)
                # plt.imshow(target, cmap='gray')
                # plt.subplot(2,2,2)
                # plt.imshow(preds_2, cmap='gray')
                # plt.subplot(2,2,3)
                # plt.imshow(im_bin_, cmap='gray')
                # plt.subplot(2,2,4)
                # plt.imshow(img_bbox)
                # Ensure that the image/mask to be saved is not in the list of empty masks
                count = [name in empty_mask[j] for j in range(len(empty_mask))].count(True)
                if count == 0:
                    imsave(f"{FOLDER_SAVE_PREDICT}/{name}_seg.png", im_bin_)        
                    if OPT == 0:
                        imsave(f"{FOLDER_SAVE_CROPED}/croped_{name}.jpg", img_bbox)        

                # actual = target.flatten()
                # predicted = im_bin_.flatten()
                # # We count the pairs (actual, predict). histogram2d actually counts x-y pairs
                # actual_predicted, _, _ = np.histogram2d(actual, predicted,
                #                     bins=nb_labels, range=[(0, nb_labels), (0, nb_labels)])
                # _actual_predicted += np.int64(actual_predicted)

            # TP, TN, FP_A, FP_B, FN = _get_global_confusion_items(_actual_predicted)
            # items = ConfusionItems(TP, TN, FP_A + FP_B, FN)
            # results = compute_confusion_results(items)
            # results_seg = _compute_segment_metrics(TP, TN, FP_A, FP_B, FN)
            # print(f"Got the results: {results_seg}")

            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
            print(f"IOU score: {IOU_score/len(data_loader)}")
            print(f"IOU score without postprocessing: {IOU_score_wout/len(data_loader)}")
if __name__ == "__main__":
    main()