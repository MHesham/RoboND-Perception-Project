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
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

VOXEL_LEAF_SIZE = 0.01
PICK_PLACE_ENABLE = True
TEST_SCENE_NUM = 1
DEBUG_SVM = True
DEBUG_PERCEPTION = False
PR2_STATE_MACHINE_ENABLE = True
PR2_WORLD_JOINT_IDX = 19
PR2_ROTATE_LEFT_ANGLE = 1.35
PR2_ROTATE_RIGHT_ANGLE = -1.35
PR2_ROTATE_CENTER_ANGLE = 0.


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


def get_pcl_centroid(pcl_cloud):
    # Get the PointCloud for a given object and obtain it's centroid
    # Convert ROS point clound to a 2D numpy float32 array where each row is [x,y,z,rgb]
    points_arr = pcl_cloud.to_array()
    # Compute the mean along the 1st dimension and extract the first 3 elements which
    # correspond to [x,y,z]
    centroid = np.mean(points_arr, axis=0)[0:3]
    return [np.asscalar(centroid[0]),
            np.asscalar(centroid[1]),
            np.asscalar(centroid[2])]


class Pr2Perception:
    def __init__(self):
        # Create publishers
        self.pcl_objects_pub = rospy.Publisher(
            "/pcl_objects", PointCloud2, queue_size=1)
        self.pcl_stat_outlier_removal_pub = rospy.Publisher(
            '/stat_outlier_removal', PointCloud2, queue_size=1)
        self.pcl_table_pub = rospy.Publisher(
            "/pcl_table", PointCloud2, queue_size=1)
        self.pcl_cluster_pub = rospy.Publisher(
            "/pcl_cluster", PointCloud2, queue_size=1)
        self.object_markers_pub = rospy.Publisher(
            "/object_markers", Marker, queue_size=1)
        self.detected_objects_pub = rospy.Publisher(
            "/detected_objects", DetectedObjectsArray, queue_size=1)

        self.pcl_cloud = None
        self.detected_objects_list = []
        self.ros_cloud_table = None

        # Load Model From disk
        model = pickle.load(open('model.sav', 'rb'))
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']

    def update(self, pcl_cloud):
        self.pcl_cloud = pcl_cloud
        self.filtering()
        cluster_indices, pcl_cloud_objects = self.segmentation()
        self.object_detection(cluster_indices, pcl_cloud_objects)

    def filtering(self):
        time_start = time()

        # 1st PassThrough Filter to extract points on the table level
        passthrough_fltr = self.pcl_cloud.make_passthrough_filter()
        filter_axis = 'z'
        passthrough_fltr.set_filter_field_name(filter_axis)
        axis_min = 0.6
        axis_max = 1.1
        passthrough_fltr.set_filter_limits(axis_min, axis_max)
        self.pcl_cloud = passthrough_fltr.filter()

        # 2nd PassThrough Filter to extract points directly infront of the robot
        passthrough_fltr = self.pcl_cloud.make_passthrough_filter()
        filter_axis = 'y'
        passthrough_fltr.set_filter_field_name(filter_axis)
        axis_min = -0.5
        axis_max = 0.5
        passthrough_fltr.set_filter_limits(axis_min, axis_max)
        self.pcl_cloud = passthrough_fltr.filter()

        # 1.1. Statistical Outlier Filtering
        outlier_fltr = self.pcl_cloud.make_statistical_outlier_filter()
        # Set the number of neighboring points to analyze for any given point
        outlier_fltr.set_mean_k(5)
        # Set threshold scale factor
        std_mul = 1
        # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
        outlier_fltr.set_std_dev_mul_thresh(std_mul)
        self.pcl_cloud = outlier_fltr.filter()

        ros_cloud = pcl_to_ros(self.pcl_cloud)
        self.pcl_stat_outlier_removal_pub.publish(ros_cloud)

        # 1.2. Voxel Grid Downsampling
        vox_fltr = self.pcl_cloud.make_voxel_grid_filter()
        vox_fltr.set_leaf_size(
            VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE)
        self.pcl_cloud = vox_fltr.filter()

        rospy.loginfo('filtering elapsed time: {}s'.format(
            time() - time_start))

    def segmentation(self):
        time_start = time()

        # 2.1. RANSAC Plane Segmentation
        seg = self.pcl_cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_model_type(pcl.SAC_RANSAC)
        max_distance = 0.01
        seg.set_distance_threshold(max_distance)
        inliers, coefficients = seg.segment()

        # 2.2. Extract plane inliers and outliers
        cloud_objects = self.pcl_cloud.extract(inliers, negative=True)
        cloud_table = self.pcl_cloud.extract(inliers, negative=False)

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
        self.ros_cloud_table = pcl_to_ros(cloud_table)
        ros_cloud_cluster = pcl_to_ros(cloud_cluster)

        # Publish ROS messages
        self.pcl_objects_pub.publish(ros_cloud_objects)
        self.pcl_table_pub.publish(self.ros_cloud_table)
        self.pcl_cluster_pub.publish(ros_cloud_cluster)

        rospy.loginfo('segmentation elapsed time: {}s'.format(
            time() - time_start))

        return cluster_indices, cloud_objects

    def object_detection(self, cluster_indices, cloud_objects):
        time_start = time()
        do_labels = []
        self.detected_objects_list = []

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
            # and add it to do_labels list
            prediction = self.clf.predict(
                self.scaler.transform(feature.reshape(1, -1)))
            rospy.loginfo('Prediction: {} {}'.format(prediction,
                                                     self.encoder.inverse_transform(prediction)))
            label = self.encoder.inverse_transform(prediction)[0]
            do_labels.append(label)

            if DEBUG_SVM is True:
                prediction_proba = self.clf.predict_proba(
                    self.scaler.transform(feature.reshape(1, -1)))
                rospy.loginfo(
                    'Prediction probabilities: {} for classes {}'.format(prediction_proba, self.encoder.classes_))
                probability = np.max(prediction_proba)
                rospy.loginfo(
                    'Detected {} with probability {}'.format(label, probability))

            # Publish a label into RViz
            label_pos = get_pcl_centroid(pcl_cluster)
            label_pos[2] += .1
            marker_label = '{} {:0.2f}'.format(label, np.asscalar(probability))
            self.object_markers_pub.publish(
                make_label(marker_label, label_pos, index))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = '{}'.format(label)
            do.cloud = ros_cluster
            self.detected_objects_list.append(do)

        rospy.loginfo('object detection elapsed time: {}s'.format(
            time() - time_start))

        # Publish the list of detected objects
        rospy.loginfo('Detected {} objects: {}'.format(
            len(do_labels), do_labels))
        self.detected_objects_pub.publish(self.detected_objects_list)

    def get_detected_object_by_label(self, label):
        for detected_object in self.detected_objects_list:
            if detected_object.label == label:
                return detected_object
        return None


class PickPlaceSrvParam:
    def __init__(self):
        self.test_scene_num = Int32()
        self.object_name = String()
        self.arm_name = String()
        self.pick_pose = Pose()
        self.place_pose = Pose()


class Pr2StateMachine:
    PR2_STATE_INIT = 'init'
    PR2_STATE_IDLE = 'idle'
    PR2_STATE_EXPLORE_RIGHT = 'right'
    PR2_STATE_EXPLORE_LEFT = 'left'
    PR2_STATE_PICK_PLACE = 'pick_place'

    def __init__(self, perception):
        # Create publishers
        self.collision_points_pub = rospy.Publisher(
            "/pr2/3d_map/points", PointCloud2, queue_size=1)
        self.joint_controller_pub = rospy.Publisher(
            "/pr2/world_joint_controller/command", Float64, queue_size=1)

        self.pcl_left_table_pub = rospy.Publisher(
            "/pcl_left_table", PointCloud2, queue_size=1)

        self.pcl_right_table_pub = rospy.Publisher(
            "/pcl_right_table", PointCloud2, queue_size=1)

        self.state = self.PR2_STATE_INIT
        self.perception = perception

        self.ros_cloud_left_table = None
        self.ros_cloud_right_talbe = None

        # Createy Service Proxies
        rospy.wait_for_service('clear_octomap')
        self.clear_octmap_srv = rospy.ServiceProxy('clear_octomap', Empty)

        self.pick_object_list_param = rospy.get_param('/object_list')
        self.picked_objects_set = set()

        dropbox_param = rospy.get_param('/dropbox')
        self.dropbox_dict = {}
        for i in range(len(dropbox_param)):
            name = dropbox_param[i]['name']
            group = dropbox_param[i]['group']
            pos = dropbox_param[i]['position']
            self.dropbox_dict[group] = {'name': name, 'position': pos}

    def update(self):
        rospy.loginfo('PR2 current state is {}'.format(self.state))
        previous_state = self.state
        if self.state == self.PR2_STATE_INIT:
            self.update_init()
        if self.state == self.PR2_STATE_IDLE:
            self.update_idle()
        elif self.state == self.PR2_STATE_EXPLORE_LEFT:
            self.update_explore_left()
        elif self.state == self.PR2_STATE_EXPLORE_RIGHT:
            self.update_explore_right()
        elif self.state == self.PR2_STATE_PICK_PLACE:
            self.update_pick_place()
        else:
            raise ValueError('undefined PR2 state')

        # Log rotation state change.
        if previous_state != self.state:
            rospy.loginfo(
                'PR2 state updated {} -> {}'.format(previous_state, self.state))

    def clear_collision_map(self):
        try:
            self.clear_octmap_srv()
        except rospy.ServiceException, e:
            rospy.logerr('Service call failed: {}'.format(e))

    def update_init(self):
        self.clear_collision_map()
        self.rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
        self.state = self.PR2_STATE_EXPLORE_LEFT

    def update_explore_left(self):
        if not self.rotate_at_goal(PR2_ROTATE_LEFT_ANGLE):
            self.rotate_in_place(PR2_ROTATE_LEFT_ANGLE)
        else:
            assert self.rotate_at_goal(PR2_ROTATE_LEFT_ANGLE)
            assert self.perception.ros_cloud_table is not None
            rospy.loginfo('Capturing left table cloud')
            self.ros_cloud_left_table = self.perception.ros_cloud_table
            self.rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
            self.state = self.PR2_STATE_EXPLORE_RIGHT

    def update_explore_right(self):
        if not self.rotate_at_goal(PR2_ROTATE_RIGHT_ANGLE):
            self.rotate_in_place(PR2_ROTATE_RIGHT_ANGLE)
        else:
            assert self.rotate_at_goal(PR2_ROTATE_RIGHT_ANGLE)
            assert self.perception.ros_cloud_table is not None
            rospy.loginfo('Capturing right table cloud')
            self.ros_cloud_right_table = self.perception.ros_cloud_table
            self.rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
            self.state = self.PR2_STATE_IDLE

    def update_idle(self):
        if self.can_pick_object():
            self.state = self.PR2_STATE_PICK_PLACE

    def update_pick_place(self):
        assert self.can_pick_object()
        self.clear_collision_map()
        assert self.ros_cloud_left_table is not None
        self.pcl_left_table_pub.publish(self.ros_cloud_left_table)
        self.collision_points_pub.publish(self.ros_cloud_left_table)
        assert self.ros_cloud_right_table is not None
        self.pcl_right_table_pub.publish(self.ros_cloud_right_table)
        self.collision_points_pub.publish(self.ros_cloud_right_table)
        assert self.perception.ros_cloud_table is not None
        self.collision_points_pub.publish(self.perception.ros_cloud_table)

        try:
            self.pick_place_object()
        except rospy.ROSInterruptException:
            pass
        self.state = self.PR2_STATE_IDLE

    def rotate_at_goal(self, goal_j1):
        joint_state = rospy.wait_for_message(
            '/pr2/joint_states', JointState)
        curr_j1 = joint_state.position[PR2_WORLD_JOINT_IDX]
        return Pr2StateMachine.rotate_at_goal_helper(curr_j1, goal_j1)

    @staticmethod
    def rotate_at_goal_helper(pos_j1, goal_j1):
        tolerance = .01
        result = abs(pos_j1 - goal_j1) <= abs(tolerance)
        return result

    def rotate_in_place(self, pos_j1):
        time_elapsed = rospy.Time.now()
        self.joint_controller_pub.publish(pos_j1)
        while True:
            joint_state = rospy.wait_for_message(
                '/pr2/joint_states', JointState)

            curr_j1 = joint_state.position[PR2_WORLD_JOINT_IDX]
            if Pr2StateMachine.rotate_at_goal_helper(curr_j1, pos_j1):
                time_elapsed = joint_state.header.stamp - time_elapsed
                break

        rospy.loginfo('PR2 world_joint rotated to {} in {}s'.format(
            pos_j1, time_elapsed.to_sec()))

    def can_pick_object(self):
        for i in range(len(self.pick_object_list_param)):
            pick_object_name = self.pick_object_list_param[i]['name']
            if pick_object_name not in self.picked_objects_set:
                do = self.perception.get_detected_object_by_label(
                    pick_object_name)
                if do is not None:
                    return True
        return False

    def pick_place_object(self):
        yaml_dict_list = []
        target_pick_place_param = PickPlaceSrvParam()
        target_object_found = False

        # Publish cloud of left, right and center tables to construct the
        # collision map from scratch by clearing it first.
        # Loop through the pick list and pick the first non-picked item
        for i in range(len(self.pick_object_list_param)):
            pick_object_name = self.pick_object_list_param[i]['name']

            if pick_object_name in self.picked_objects_set:
                continue

            do = self.perception.get_detected_object_by_label(pick_object_name)
            if do is None:
                rospy.logerr('Object {} not found in detected objects'.format(
                    pick_object_name))
                continue

            pick_place_param = PickPlaceSrvParam()
            # Create 'place_pose' for the object
            pick_place_param.test_scene_num.data = TEST_SCENE_NUM

            pick_place_param.object_name.data = pick_object_name

            # Set object pick pose
            centroid = get_pcl_centroid(ros_to_pcl(do.cloud))
            pick_place_param.pick_pose.position.x = centroid[0]
            pick_place_param.pick_pose.position.y = centroid[1]
            pick_place_param.pick_pose.position.z = centroid[2]

            # Set object place pose
            pick_object_group = self.pick_object_list_param[i]['group']
            dropbox_pos = self.dropbox_dict[pick_object_group]['position']
            pick_place_param.place_pose.position.x = dropbox_pos[0]
            pick_place_param.place_pose.position.y = dropbox_pos[1]
            pick_place_param.place_pose.position.z = dropbox_pos[2]

            # Assign the arm to be used for pick_place_object. Valid values: right, left.
            pick_place_param.arm_name.data = self.dropbox_dict[pick_object_group]['name']

            # Create a list of dictionaries (made with make_yaml_dict()) for later
            # output to yaml format
            yaml_dict = make_yaml_dict(
                pick_place_param.test_scene_num,
                pick_place_param.arm_name,
                pick_place_param.object_name,
                pick_place_param.pick_pose,
                pick_place_param.place_pose)
            yaml_dict_list.append(yaml_dict)

            if (target_object_found is False) and \
                    (pick_object_name not in self.picked_objects_set):
                target_object_found = True
                target_pick_place_param = pick_place_param
            else:
                self.collision_points_pub.publish(do.cloud)

        # Output request parameters into output yaml file
        send_to_yaml('output_{}.yaml'.format(TEST_SCENE_NUM), yaml_dict_list)

        if (target_object_found is True) and PICK_PLACE_ENABLE:
            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')

            try:
                pick_place_routine = rospy.ServiceProxy(
                    'pick_place_routine', PickPlace)

                assert target_pick_place_param.object_name not in self.picked_objects_set
                resp = pick_place_routine(
                    target_pick_place_param.test_scene_num,
                    target_pick_place_param.object_name,
                    target_pick_place_param.arm_name,
                    target_pick_place_param.pick_pose,
                    target_pick_place_param.place_pose)

                rospy.loginfo('Response: {}'.format(resp.success))
                if resp.success is True:
                    self.picked_objects_set.add(
                        target_pick_place_param.object_name)

            except rospy.ServiceException, e:
                rospy.logerr('Service call failed: {}'.format(e))


skip_frame = False


def pcl_callback(pcl_msg):
    global skip_frame
    rospy.loginfo('### PCL BEGIN ###')

    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    time_start = time()

    pr2_perception.update(pcl_cloud)
    if PR2_STATE_MACHINE_ENABLE:
        if not skip_frame:
            pr2_state_machine.update()
            skip_frame = True
        else:
            skip_frame = False

    rospy.loginfo('### PCL END ## elapsed time: {}s'.format(
        time() - time_start))

# function to load parameters and request PickPlace service


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('pr2_perception', anonymous=True)

    # Initialize color_list
    get_color_list.color_list = []

    pr2_perception = Pr2Perception()
    pr2_state_machine = Pr2StateMachine(pr2_perception)

    # Create Subscribers
    pcl_sub = rospy.Subscriber(
        "/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
