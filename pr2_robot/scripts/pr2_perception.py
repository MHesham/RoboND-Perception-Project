#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from pcl_helper import *
from time import time

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

VOXEL_LEAF_SIZE = 0.01

# Helper function to get surface normals
PICK_PLACE_ENABLE = False
TEST_SCENE_NUM = 1
DEBUG_SVM = True


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages


def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict

# Helper function to output to yaml file


def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def filtering(pcl_cloud):

    time_start = time()

    # 1.1. Statistical Outlier Filtering
    outlier_fltr = pcl_cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_fltr.set_mean_k(10)
    # Set threshold scale factor
    std_mul = .05
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_fltr.set_std_dev_mul_thresh(std_mul)
    pcl_cloud = outlier_fltr.filter()
    ros_cloud = pcl_to_ros(pcl_cloud)

    pcl_stat_outlier_removal_pub.publish(ros_cloud)

    # 1.2. Voxel Grid Downsampling
    vox_fltr = pcl_cloud.make_voxel_grid_filter()
    vox_fltr.set_leaf_size(VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE)
    pcl_cloud = vox_fltr.filter()

    # 1.3. PassThrough Filter
    passthrough_fltr = pcl_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_fltr.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_fltr.set_filter_limits(axis_min, axis_max)
    pcl_cloud = passthrough_fltr.filter()

    passthrough_fltr = pcl_cloud.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_fltr.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough_fltr.set_filter_limits(axis_min, axis_max)
    pcl_cloud = passthrough_fltr.filter()

    rospy.loginfo('filtering elapsed time: {}s'.format(time() - time_start))

    return pcl_cloud


def segmentation(pcl_cloud):
    time_start = time()

    # 2.1. RANSAC Plane Segmentation
    seg = pcl_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_model_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # 2.2. Extract plane inliers and outliers
    cloud_objects = pcl_cloud.extract(inliers, negative=True)
    cloud_table = pcl_cloud.extract(inliers, negative=False)

    #
    # 2.3. Clustering for Segmentation
    #

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(3000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    rospy.loginfo("cluster indices count %d", len(cluster_indices))
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append(
                [white_cloud[index][0], white_cloud[index][1], white_cloud[index][2], rgb_to_float(cluster_color[j])])

    cloud_cluster = pcl.PointCloud_PointXYZRGB()
    cloud_cluster.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_cluster = pcl_to_ros(cloud_cluster)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)

    rospy.loginfo('segmentation elapsed time: {}s'.format(time() - time_start))

    return cluster_indices, cloud_objects, white_cloud


def object_detection(cluster_indices, cloud_objects, white_cloud_objects):
    time_start = time()
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        rospy.loginfo('Prediction: {} {}'.format(prediction,
                                                 encoder.inverse_transform(prediction)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        if DEBUG_SVM is True:
            prediction_proba = clf.predict_proba(
                scaler.transform(feature.reshape(1, -1)))
            rospy.loginfo(
                'Prediction probabilities: {}'.format(prediction_proba))

        # Publish a label into RViz
        x_sum, y_sum, z_sum = 0., 0., 0.
        for pidx in pts_list:
            pos = list(white_cloud_objects[pidx])
            x_sum += pos[0]
            y_sum += pos[1]
            z_sum += pos[2]

        n = len(pts_list)
        label_pos = [x_sum / n, y_sum / n, z_sum / n]
        label_pos[2] += .1
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('object detection elapsed time: {}s'.format(
        time() - time_start))

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(
        len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    return detected_objects

# Callback function for your Point Cloud Subscriber


def pcl_callback(pcl_msg):

    rospy.loginfo('received a pcl')
    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    time_start = time()

    #
    # 1. Filtering
    #
    pcl_cloud = filtering(pcl_cloud)

    #
    # 2. Segmentation
    #
    cluster_indices, cloud_objects, white_cloud = segmentation(pcl_cloud)

    #
    # 3. Object Detection
    #
    detected_objects_list = object_detection(
        cluster_indices, cloud_objects, white_cloud)

    rospy.loginfo('perception pipeline elapsed time: {}s'.format(
        time() - time_start))

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service


def get_detected_object_by_label(object_list, label):
    for detected_object in object_list:
        if detected_object.label == label:
            return detected_object
    return None


def get_dropbox_dict():
    dropbox_param = rospy.get_param('/dropbox')
    dropbox_dict = {}
    for i in range(len(dropbox_param)):
        name = dropbox_param[i]['name']
        group = dropbox_param[i]['group']
        pos = dropbox_param[i]['position']
        dropbox_dict[group] = {'name': name, 'position': pos}
    return dropbox_dict


def pr2_mover(detected_object_list):

    yaml_dict_list = []
    pick_object_list_param = rospy.get_param('/object_list')
    dropbox_dict = get_dropbox_dict()

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for i in range(len(pick_object_list_param)):
        pick_object_name = pick_object_list_param[i]['name']
        pick_object_group = pick_object_list_param[i]['group']

        do = get_detected_object_by_label(
            detected_object_list, pick_object_name)
        if do is None:
            rospy.logerr('Object {} not found in detected objects {}'.format(
                pick_object_name, detected_object_list))
            continue

        # Get the PointCloud for a given object and obtain it's centroid
        # Convert ROS point clound to a 2D numpy float32 array where each row is [x,y,z,rgb]
        points_arr = ros_to_pcl(do.cloud).to_array()
        # Compute the mean along the 1st dimension and extract the first 3 elements which
        # correspond to [x,y,z]
        centroid = np.mean(points_arr, axis=0)[0:3]

        # Create 'place_pose' for the object
        req_test_scene_num = Int32()
        req_test_scene_num.data = TEST_SCENE_NUM

        req_object_name = String()
        req_object_name.data = pick_object_name

        # Set object pick pose
        req_pick_pose = Pose()
        req_pick_pose.position.x = np.asscalar(centroid[0])
        req_pick_pose.position.y = np.asscalar(centroid[1])
        req_pick_pose.position.z = np.asscalar(centroid[2])

        # Set object place pose
        dropbox_pos = dropbox_dict[pick_object_group]['position']
        req_place_pose = Pose()
        req_place_pose.position.x = dropbox_pos[0]
        req_place_pose.position.y = dropbox_pos[1]
        req_place_pose.position.z = dropbox_pos[2]

        # Assign the arm to be used for pick_place. Valid values: right, left.
        req_arm_name = String()
        req_arm_name.data = dropbox_dict[pick_object_group]['name']

        # Create a list of dictionaries (made with make_yaml_dict()) for later
        # output to yaml format
        yaml_dict = make_yaml_dict(
            req_test_scene_num, req_arm_name, req_object_name, req_pick_pose, req_place_pose)
        yaml_dict_list.append(yaml_dict)

        if PICK_PLACE_ENABLE is True:
            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')

            try:
                pick_place_routine = rospy.ServiceProxy(
                    'pick_place_routine', PickPlace)

                resp = pick_place_routine(
                    test_scene_num, object_name, arm_name, pick_pose, place_pose)

                rospy.loginfo('Response: {}'.format(resp.success))

            except rospy.ServiceException, e:
                rospy.loginfo('Service call failed: {}'.format(e))

    # Output request parameters into output yaml file
    send_to_yaml('output_{}.yaml'.format(TEST_SCENE_NUM), yaml_dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('pr2_perception', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber(
        "/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher(
        "/pcl_objects", PointCloud2, queue_size=1)
    pcl_stat_outlier_removal_pub = rospy.Publisher(
        '/stat_outlier_removal', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher(
        "/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher(
        "/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher(
        "/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
