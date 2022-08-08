# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:24:52 2022

@author: adejunior
"""


import open3d as o3d
import laspy
import numpy as np
import pandas as pd
# from laspy.file import File
from scipy.spatial.kdtree import KDTree
# from sklearn.decomposition import PCA
from skimage.exposure import cumulative_distribution
# \from pyntcloud import PyntCloud
from matplotlib.pyplot import plt

from time import time



def load_pcd(file_address, sep=' ', header=None, skiprows=None):
    df = pd.read_csv(file_address, sep=sep, delimiter=None, header=header, skiprows=None)
    xyz_load = np.concatenate((np.reshape(np.asarray(df[0]), (-1,1)),
                               np.reshape(np.asarray(df[1]), (-1,1)),
                               np.reshape(np.asarray(df[2]), (-1,1))), axis=1)
    
    colors = np.concatenate((np.reshape(np.asarray(df[3]), (-1,1)),
                             np.reshape(np.asarray(df[4]), (-1,1)),
                             np.reshape(np.asarray(df[5]), (-1,1))), axis=1)
    
    return xyz_load, colors


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def custom_draw_geometry_load_option(pcd, camera_settings):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_settings)
    ctr.convert_from_pinhole_camera_parameters(parameters)
    
    vis.run()
    vis.destroy_window()
    
    
def cdf(im):
    '''
    computes the CDF of an image im as 2D numpy ndarray
    '''
    c, b = cumulative_distribution(im) 
    # pad the beginning and ending pixels and their CDF values
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c

def hist_matching(c, c_t, im):
    '''
    c: CDF of input image computed with the function cdf()
    c_t: CDF of template image computed with the function cdf()
    im: input image as 2D numpy ndarray
    returns the modified pixel values
    ''' 
    pixels = np.arange(256)
    # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
    # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
    new_pixels = np.interp(c, c_t, pixels) 
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
    return im

def rmse(mean_distances):
    value = np.sqrt(np.sum(np.power(np.mean(mean_distances)-mean_distances, 2))/np.size(mean_distances))
    return value


def mean(mean_distances):
    value = np.mean(np.asarray(mean_distances)[np.int64(np.asarray(np.where(mean_distances<(np.mean(mean_distances)+np.std(mean_distances)))).tolist())])
    return value


def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1

    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            
            if np.size(np.where(cloud_index[node.indices] == 0)) > np.size(np.where(cloud_index[node.indices] == 1)):
                for points in np.asarray(node.indices)[np.where(cloud_index[node.indices] == 0)]:
                    pcd_out.append(pcd.points[points])
                    if color == True:
                        color_out.append(pcd.colors[points])
                    else:
                        color_out.append((0, 0, 1))
            else:
                for points in np.asarray(node.indices)[np.where(cloud_index[node.indices] == 1)]:
                    pcd_out.append(pcd.points[points])
                    if color == True:
                        color_out.append(pcd.colors[points])
                    else:
                        color_out.append((1, 0, 0))

    else:
        raise NotImplementedError('Node type not recognized!')

    return early_stop




camera_settings = "ScreenCamera_2022-07-04-16-03-42.json"
camera_settings = "ScreenCamera_2022-07-04-16-20-24.json"
camera_settings = "ScreenCamera_2022-07-04-16-36-30.json"


'''
Loading point clouds and offset to position
'''
# xyz_1, colors_1 = load_pcd("DenseCloud_GoPro.xyz")
# xyz_2, colors_2 = load_pcd("DenseCloud_Mavic.xyz")

xyz_1, colors_1 = load_pcd("DenseCloud_GoPro_Full_C2C.txt", ' ', None, None)
xyz_2, colors_2 = load_pcd("DenseCloud_Mavic_Full_C2C.txt", ' ', None, None)

xyz_load = np.concatenate((xyz_1, xyz_2), axis=0)
colors = np.concatenate((colors_1, colors_2), axis=0)

# xyz_load, colors = load_pcd("DenseCloud_Mavic_Full_C2C.txt", ' ', None, None)



anchor_offset = [np.min(xyz_load[:, 0]), np.min(xyz_load[:, 1])]
xyz_load[:,0] = xyz_load[:,0] - anchor_offset[0]
xyz_load[:,1] = xyz_load[:,1] - anchor_offset[1] 

cloud_index = np.concatenate((np.full(np.shape(xyz_1)[0], 0),
                              np.full(np.shape(xyz_2)[0], 1)), axis=0 )


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_load)
pcd.colors = o3d.utility.Vector3dVector(colors/255)
# o3d.visualization.draw_geometries([pcd])
custom_draw_geometry_load_option(pcd, camera_settings)



'''
Join point clouds with octree segmentation
'''
octree = o3d.geometry.Octree(max_depth=7)
octree.convert_from_point_cloud(pcd, size_expand=0.01)
# o3d.visualization.draw_geometries([octree])


pcd_out = []
color_out = []
color=True
octree.traverse(f_traverse)


pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(np.asarray(pcd_out))
pcd_2.colors = o3d.utility.Vector3dVector(np.asarray(color_out))
# o3d.visualization.draw_geometries([pcd_2])
custom_draw_geometry_load_option(pcd_2, camera_settings)


o3d.io.write_point_cloud("optimized_UAV_SPI_pcd.xyz", pcd_2)


'''
Joint point cloud with outliers removed
'''
cl, ind = pcd_2.remove_statistical_outlier(nb_neighbors=10,
                                           std_ratio=2.0)
# display_inlier_outlier(pcd_2, ind)

inlier_points = np.asarray(pcd_out)[ind]
inlier_colors = np.asarray(color_out)[ind]

pcd_3 =  o3d.geometry.PointCloud()
pcd_3.points = o3d.utility.Vector3dVector(inlier_points)
pcd_3.colors = o3d.utility.Vector3dVector(inlier_colors)

# o3d.visualization.draw_geometries([pcd_3])
custom_draw_geometry_load_option(pcd_2, camera_settings)


# Equalizing colors using the one of the clouds as reference
im = hist_matching(cdf(colors_1), cdf(colors_2), colors_1)



'''
Computing distances for the Mavik point cloud
'''
start_time = time()
mean_distances = []
dataset = xyz_load
tree = KDTree(dataset)
for i in range(0, int(np.size(dataset)/3)):
    distances, points = tree.query(dataset[i,], k = 10)
    mean_distances.append(np.mean(distances))
    # pca = PCA(n_components=3)
    # pca.fit(dataset[points])
    # pca.components_
    # pca.explained_variance_
    
mean_mavik = np.mean(mean_distances)
print(time()-start_time)
    
# Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
# header.offsets = np.min(my_data, axis=0)
# header.scales = np.array([0.1, 0.1, 0.1])

# Create a LasWriter and a point record, then write it
with laspy.open("output0.las", mode="w", header=header) as writer:
    point_record = laspy.ScaleAwarePointRecord.zeros(dataset.shape[0], header=header)
    point_record.x = dataset[:, 0]
    point_record.y = dataset[:, 1]
    point_record.z = dataset[:, 2]
    point_record.intensity = np.uint16(np.asarray(mean_distances)*100)

    writer.write_points(point_record)
print(time()-start_time)
    


'''
Computing distances for the Mavik point cloud
'''
start_time = time()
mean_distances = []
dataset = xyz_2
tree = KDTree(dataset)
for i in range(0, int(np.size(dataset)/3)):
    distances, points = tree.query(dataset[i,], k = 10)
    mean_distances.append(np.mean(distances))
    # pca = PCA(n_components=3)
    # pca.fit(dataset[points])
    # pca.components_
    # pca.explained_variance_
    
mean_full = np.mean(mean_distances)
print(time()-start_time)
    
# Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
# header.offsets = np.min(my_data, axis=0)
# header.scales = np.array([0.1, 0.1, 0.1])

# Create a LasWriter and a point record, then write it
with laspy.open("output1.las", mode="w", header=header) as writer:
    point_record = laspy.ScaleAwarePointRecord.zeros(dataset.shape[0], header=header)
    point_record.x = dataset[:, 0]
    point_record.y = dataset[:, 1]
    point_record.z = dataset[:, 2]
    point_record.intensity = np.uint16(np.asarray(mean_distances)*100)

    writer.write_points(point_record)
print(time()-start_time)


'''
Computing distances for the joint point cloud
'''
start_time = time()
mean_distances_f = []
dataset = np.asarray(pcd_out)
tree = KDTree(dataset)
for i in range(0, int(np.size(dataset)/3)):
    distances, points = tree.query(dataset[i,], k = 10)
    mean_distances_f.append(np.mean(distances))
    # pca = PCA(n_components=3)
    # pca.fit(dataset[points])
    # pca.components_
    # pca.explained_variance_
    
mean_full_optimized = np.mean(mean_distances_f)
print(time()-start_time)
    
# Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
# header.offsets = np.min(my_data, axis=0)
# header.scales = np.array([0.1, 0.1, 0.1])

# Create a LasWriter and a point record, then write it
with laspy.open("output2.las", mode="w", header=header) as writer:
    point_record = laspy.ScaleAwarePointRecord.zeros(dataset.shape[0], header=header)
    point_record.x = dataset[:, 0]
    point_record.y = dataset[:, 1]
    point_record.z = dataset[:, 2]
    point_record.intensity = np.uint16(np.asarray(mean_distances_f)*100)

    writer.write_points(point_record)
print(time()-start_time)


'''
Statistics on point distances
'''
mean(mean_distances)
mean(mean_distances_f)

rmse(mean_distances)
rmse(mean_distances_f)


a = np.empty((np.size(mean_distances_f)-np.size(mean_distances)))
a[:] = np.nan
a= a.tolist()

b = [*mean_distances, *a]

data = np.vstack((np.asarray(b), np.asarray(mean_distances_f)))

df = pd.DataFrame(data.T,
                  columns=['UAV/SPI', 'UAV/SPI optimized'])
boxplot = df.boxplot(column=['UAV/SPI', 'UAV/SPI optimized'])



# cloud = PyntCloud(pd.DataFrame(
#     # same arguments that you are passing to visualize_pcl
#     data=np.hstack((dataset, np.reshape(np.asarray(mean_distances), (-1,1))   )),
#     columns=["x", "y", "z", "mean_distances"]))

# cloud.to_file("output2.ply")
    
    
# # From the KNN selection of n points, the studied metrics observe the 
# # geometric distances and curvature using quadratic fitting (Chetouani et al, 2021), entropy,
# # generalized gaussian distribution, assymetric generalized gaussian
# # distribution, and gamma as 3D Natural Scene Statistics (Zhang et al, 2022),
# # composed using regression (Zhang et al, 2022) or convolution neural network
# # (Chetouani et al, 2021). Liu et al 2021 and Yang et al 2021 use trained CNNs
# # directly to the point cloud data to estimate point cloud quality.
