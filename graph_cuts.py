import maxflow
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from sklearn.cluster import KMeans


def segment_1(im , mean_object_dark, mean_object_clear, mean_background, p=1, beta = 100):

    if p == 1:
        m0 = mean_object_dark
    else:
        m0 = mean_object_clear
    m1 = mean_background

    ## Graph cut binaire
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(im.shape[:2])
    g.add_grid_edges(nodeids, beta)
    g.add_grid_tedges(nodeids, np.sqrt((im[:,:,0]-m0[0])**2 + (im[:,:,1]-m0[1])**2 + (im[:,:,2]-m0[2])**2), np.sqrt((im[:,:,0]-m1[0])**2 + (im[:,:,1]-m1[1])**2 + (im[:,:,2]-m1[2])**2))
    flow = g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    im_bin = 1 - np.int_(np.logical_not(sgm))
    return im_bin

def clustering(im, im_bin):
    X_1 = np.vstack([im[im_bin==1,i].flatten() for i in range(3)]).transpose()
    kmeans_lesion2 = KMeans(n_clusters=5, random_state=0).fit(X_1)
    mean_vectors_lesion2 = kmeans_lesion2.cluster_centers_

    X_2 = np.vstack([im[im_bin==0,i].flatten() for i in range(3)]).transpose()
    kmeans_background2 = KMeans(n_clusters=5, random_state=0).fit(X_2)
    mean_vectors_background2 = kmeans_background2.cluster_centers_
    neg_log_likelihood_lesion_combined_2 = np.amin(np.dstack([(sum((im[:,:,i]-mean_vectors_lesion2[n_cl,i])**2 for i in range(3))) for n_cl in range(len(mean_vectors_lesion2))]),2)

    neg_log_likelihood_background_combined_2 = np.amin(np.dstack([(sum((im[:,:,i]-mean_vectors_background2[n_cl,i])**2 for i in range(3))) for n_cl in range(len(mean_vectors_background2))]),2)

    return neg_log_likelihood_lesion_combined_2, neg_log_likelihood_background_combined_2

def segment_2(im, neg_log_bg, neg_log_obj, beta = 1000):

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
