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
import random
import os

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

# When enabled, print extra information about the SVM classification decision
# making and the propabilities for each classified object.
# NOTE: The SVC should be created during training with probability=True
DEBUG_SVM = False

# Should match the launch file test scene number.
TEST_SCENE_NUM = 1

# If True, enables the PR2 motion parto fo the PR2 state-machine which is
# required for the perception challenge and the PR2 motion, False otherwise.
PR2_MOTION_ENABLE = True

# The index of the PR2 world joint in the joint_states. Used for rotating the
# PR2 in-place. The index was figure out by dumping the pr2/joint_states msg.
PR2_WORLD_JOINT_IDX = 19

# The angle in rad for the PR2 left rotation.
PR2_ROTATE_LEFT_ANGLE = 1.20

# The angle in rad for the PR2 right rotation.
PR2_ROTATE_RIGHT_ANGLE = -1.20

# The angle in rad for the PR2 neutral pose.
PR2_ROTATE_CENTER_ANGLE = 0.

# Dropbox width and depth manually estimated by RViz using the Measure tool.
DROPBOX_WIDTH = 0.28
DROPBOX_DEPTH = 0.76


def get_normals(cloud):
    """Get surface normals for a point cloud.

    Args:
        cloud: ROS point cloud data to get normals for.
    """
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    """Create a yaml friendly dictionary from ROS messages."""
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict


def send_to_yaml(yaml_filename, dict_list):
    """Write a list of dictionaries to a yaml file."""
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def get_pcl_centroid(pcl_cloud):
    """Return the centroid of a PCL cloud."""
    # Get the PointCloud for a given object and obtain it's centroid
    # Convert ROS point cloud to a 2D numpy float32 array where each row is [x,y,z,rgb]
    points_arr = pcl_cloud.to_array()
    # Compute the mean along the 1st dimension and extract the first 3 elements which
    # correspond to [x,y,z]
    centroid = np.mean(points_arr, axis=0)[0:3]
    return [np.asscalar(centroid[0]),
            np.asscalar(centroid[1]),
            np.asscalar(centroid[2])]


class Pr2Perception:
    """Implementation of the PR2 perception pipeline.

    The class can be looked at as a black box that takes PCL cloud as input and
    Detected Objects as output. The perception pipeline starts with filtering,
    segmentation and finally object detection. Everytime a new PCL cloud arrives
    the update method should be called with that new PCL cloud to go through
    the perception pipeline. At anytime, the detected objects can be retrieved
    through the detected_objects_list field.
    """

    # The Voxel grid volume cube side size in meters. The smaller the value
    # the larger details are perserved, and the opposite otherwise.
    VOXEL_LEAF_SIZE = 0.005

    def __init__(self):
        # Create publishers
        self.pcl_objects_pub = rospy.Publisher(
            "/pcl_objects", PointCloud2, queue_size=1)
        self.passthrough_fltr_pub = rospy.Publisher(
            '/passthrough_fltr_output', PointCloud2, queue_size=1)
        self.stat_outlier_fltr_pub = rospy.Publisher(
            '/stat_outlier_fltr_output', PointCloud2, queue_size=1)
        self.voxel_fltr_pub = rospy.Publisher(
            "/voxel_fltr_output", PointCloud2, queue_size=1)
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

        # Load Model from disk given its path supplied through a ros param.
        model_filename = rospy.get_param('~pr2_perception_model')
        model = pickle.load(open(model_filename, 'rb'))
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']

    def update(self, pcl_cloud):
        """Update the perception pipeline with a new PCL cloud."""
        self.pcl_cloud = pcl_cloud
        t0 = rospy.get_time()
        self.__filtering()
        t1 = rospy.get_time()
        cluster_indices, pcl_cloud_objects = self.__segmentation()
        t2 = rospy.get_time()
        self.__object_detection(cluster_indices, pcl_cloud_objects)
        t3 = rospy.get_time()
        rospy.loginfo('perception fltr: {}, segm: {}, objd: {}, total {}'.format(
            t1 - t0, t2 - t1, t3 - t2, t3 - t0))

    def __filtering(self):
        """Perform the filtering step of the perception pipeline."""
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

        # Publish the result of applying passthrough filter on the z and y axes
        self.passthrough_fltr_pub.publish(pcl_to_ros(self.pcl_cloud))

        # Statistical Outlier Filtering
        outlier_fltr = self.pcl_cloud.make_statistical_outlier_filter()
        # Set the number of neighboring points to analyze for any given point
        outlier_fltr.set_mean_k(5)
        # Set threshold scale factor
        std_mul = 1
        # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
        outlier_fltr.set_std_dev_mul_thresh(std_mul)
        self.pcl_cloud = outlier_fltr.filter()

        # Publish the result of removing point cloud noise
        self.stat_outlier_fltr_pub.publish(pcl_to_ros(self.pcl_cloud))

        # Voxel Grid Downsampling
        vox_fltr = self.pcl_cloud.make_voxel_grid_filter()
        vox_fltr.set_leaf_size(
            self.VOXEL_LEAF_SIZE, self.VOXEL_LEAF_SIZE, self.VOXEL_LEAF_SIZE)
        self.pcl_cloud = vox_fltr.filter()

        # Publish the result of down-sampling the point cloud
        self.voxel_fltr_pub.publish(pcl_to_ros(self.pcl_cloud))

    def __segmentation(self):
        """Perform the segmentation step of the perception pipeline"""
        # RANSAC Plane Segmentation to fit the data in a plane model to separate
        # the table and the objects.
        seg = self.pcl_cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_model_type(pcl.SAC_RANSAC)
        max_distance = 0.01
        seg.set_distance_threshold(max_distance)
        inliers, coefficients = seg.segment()

        # Extract plane inliers and outliers
        cloud_objects = self.pcl_cloud.extract(inliers, negative=True)
        cloud_table = self.pcl_cloud.extract(inliers, negative=False)

        # Euclidean Clustering (DBSCAN) for segmentation
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

        return cluster_indices, cloud_objects

    def __object_detection(self, cluster_indices, cloud_objects):
        """Detect objects given PCL cloud objects and the cluster indices."""
        do_labels = []
        del self.detected_objects_list[:]

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

            if DEBUG_SVM:
                # NOTE: The SVC should be created during training with probability=True
                prediction_proba = self.clf.predict_proba(
                    self.scaler.transform(feature.reshape(1, -1)))
                rospy.loginfo(
                    'Prediction probabilities: {} for classes {}'.format(prediction_proba, self.encoder.classes_))
                probability = np.max(prediction_proba)
                rospy.loginfo(
                    'Detected {} with probability {}'.format(label, probability))
                # The object marker label in RViz will have the object detected
                # name and its probability
                marker_label = '{} {:0.2f}'.format(
                    label, np.asscalar(probability))
            else:
                marker_label = label

            # Publish the detected object label to RViz
            label_pos = get_pcl_centroid(pcl_cluster)
            label_pos[2] += .2
            self.object_markers_pub.publish(
                make_label(marker_label, label_pos, index))

            # Add the detected object to the list of detected objects so it can
            # be retrieved later by the PR2 logic componenets
            do = DetectedObject()
            do.label = '{}'.format(label)
            do.cloud = ros_cluster
            self.detected_objects_list.append(do)

        # Publish the list of detected objects for debugging purposes.
        rospy.loginfo('Detected {} objects: {}'.format(
            len(do_labels), do_labels))
        self.detected_objects_pub.publish(self.detected_objects_list)

    def get_detected_object_by_label(self, label):
        """Find a detected object by label and return it.

        Returns:
            DetectedObject if object found, and None otherwise.
        """
        for detected_object in self.detected_objects_list:
            if detected_object.label == label:
                return detected_object
        return None


class PickPlaceSrvParam:
    """Encapsulate the pick_place_routine parameters and a task as well."""

    def __init__(self):
        self.test_scene_num = Int32()
        self.object_name = String()
        self.arm_name = String()
        self.pick_pose = Pose()
        self.place_pose = Pose()


class Pr2StateMachine:
    """Implementation of the PR2 logic for the pick and place challenge task.

    The class implements the PR2 behavior as a finite state machine which tries
    to collect left and right table collision data, and perform pick and place
    for each object in the provided pick_list in an ordered manner. It also
    writes to desk a yaml file for the pick_place_routine params for each object
    in the pick_list.
    """

    # PR2 finite state machine states which follows the following state
    # transitions:
    # init -> explore-left: on initialization done
    # explore-left -> explore-right: on left table collision data collected
    # explore-right -> plan: on right table collision data collected
    # plan -> execute: on pick_place plan built based on detected objects
    # execute -> done: on pick_place plan has completed execution
    PR2_STATE_INIT = 'init'
    PR2_STATE_EXPLORE_RIGHT = 'explore-right'
    PR2_STATE_EXPLORE_LEFT = 'explore-left'
    PR2_STATE_PLAN = 'plan'
    PR2_STATE_EXECUTE = 'execute'
    PR2_STATE_DONE = 'done'

    def __init__(self, perception):
        # Publish 3D collision map data
        self.collision_points_pub = rospy.Publisher(
            "/pr2/3d_map/points", PointCloud2, queue_size=1)
        # Publish joint angles to control PR2 orientation
        self.joint_controller_pub = rospy.Publisher(
            "/pr2/world_joint_controller/command", Float64, queue_size=1)
        # Publish left table cloud for debugging
        self.pcl_left_table_pub = rospy.Publisher(
            "/pcl_left_table", PointCloud2, queue_size=1)
        # Publish right table cloud for debugging
        self.pcl_right_table_pub = rospy.Publisher(
            "/pcl_right_table", PointCloud2, queue_size=1)

        self.state = self.PR2_STATE_INIT
        self.perception = perception

        self.ros_cloud_left_table = None
        self.ros_cloud_right_talbe = None

        self.yaml_dict_list = []
        self.task_list = []
        self.current_task_idx = 0
        self.yam_output_written = False

        # Create a proxy for the octmap clear service to be able to clear the
        # MoveIt! collision map before each pick_place iteration
        rospy.wait_for_service('clear_octomap')
        self.clear_octmap_srv = rospy.ServiceProxy('clear_octomap', Empty)

        # Read in the ordered pick_place list
        self.pick_object_list_param = rospy.get_param('/object_list')

        # Read in the dropbox data and reorganize it in a dictionary for easy
        # and efficient data access
        dropbox_param = rospy.get_param('/dropbox')
        self.dropbox_dict = {}
        for i in range(len(dropbox_param)):
            name = dropbox_param[i]['name']
            group = dropbox_param[i]['group']
            pos = dropbox_param[i]['position']
            self.dropbox_dict[group] = {'name': name, 'position': pos}
        rospy.loginfo('dropbox_dict: {}'.format(self.dropbox_dict))

    def update(self):
        rospy.loginfo('PR2 current state is {}'.format(self.state))
        previous_state = self.state
        if self.state == self.PR2_STATE_INIT:
            self.__update_init()
        elif self.state == self.PR2_STATE_EXPLORE_LEFT:
            self.__update_explore_left()
        elif self.state == self.PR2_STATE_EXPLORE_RIGHT:
            self.__update_explore_right()
        elif self.state == self.PR2_STATE_PLAN:
            self.__update_plan()
        elif self.state == self.PR2_STATE_EXECUTE:
            self.__update_execute()
        elif self.state != self.PR2_STATE_DONE:
            raise ValueError('undefined PR2 state')

        if previous_state != self.state:
            rospy.logwarn(
                'PR2 state updated {} -> {}'.format(previous_state, self.state))

    def __clear_collision_map(self):
        """Clears MoveIt! collision map and start from scratch"""
        try:
            self.clear_octmap_srv()
        except rospy.ServiceException, e:
            rospy.logerr('Service call failed: {}'.format(e))

    def __update_init(self):
        """Update init state logic"""
        if PR2_MOTION_ENABLE:
            self.__clear_collision_map()
            self.__rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
            self.state = self.PR2_STATE_EXPLORE_LEFT
        else:
            self.state = self.PR2_STATE_PLAN

    def __update_explore_left(self):
        """Update explore-left state logic"""
        if not self.__rotate_at_goal(PR2_ROTATE_LEFT_ANGLE):
            self.__rotate_in_place(PR2_ROTATE_LEFT_ANGLE)
        else:
            assert self.__rotate_at_goal(PR2_ROTATE_LEFT_ANGLE)
            assert self.perception.ros_cloud_table is not None
            rospy.logdebug('Capturing left table cloud')
            self.ros_cloud_left_table = self.perception.ros_cloud_table
            self.__rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
            self.state = self.PR2_STATE_EXPLORE_RIGHT

    def __update_explore_right(self):
        """Update explore-right state logic"""
        if not self.__rotate_at_goal(PR2_ROTATE_RIGHT_ANGLE):
            self.__rotate_in_place(PR2_ROTATE_RIGHT_ANGLE)
        else:
            assert self.__rotate_at_goal(PR2_ROTATE_RIGHT_ANGLE)
            assert self.perception.ros_cloud_table is not None
            rospy.logdebug('Capturing right table cloud')
            self.ros_cloud_right_table = self.perception.ros_cloud_table
            self.__rotate_in_place(PR2_ROTATE_CENTER_ANGLE)
            self.state = self.PR2_STATE_PLAN

    def __update_plan(self):
        """Update plan state logic"""
        del self.task_list[:]
        self.current_task_idx = 0

        # Loop through the pick list and create a PickPlaceSrvParam for each
        # detected object and append to the task list which gets executed later
        for i in range(len(self.pick_object_list_param)):
            pick_object_name = self.pick_object_list_param[i]['name']

            do = self.perception.get_detected_object_by_label(pick_object_name)
            if do is None:
                rospy.logerr('Object {} not found in detected objects for pick operation'.format(
                    pick_object_name))
                continue

            pick_place_param = PickPlaceSrvParam()
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
            # Choose a random place in the dropbox along its depth
            place_x = random.uniform(
                dropbox_pos[0] - DROPBOX_DEPTH / 2, dropbox_pos[0])
            pick_place_param.place_pose.position.x = place_x
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
            self.yaml_dict_list.append(yaml_dict)
            self.task_list.append(pick_place_param)

        # Transition to execute state with a non-empty task list
        if len(self.task_list) > 0:
            if not self.yam_output_written:
                cwd = os.getcwd()
                rospy.logwarn('writing output_{}.yaml from {}'.format(
                    TEST_SCENE_NUM, cwd))
                # Output request parameters into output yaml file
                send_to_yaml('output_{}.yaml'.format(
                    TEST_SCENE_NUM), self.yaml_dict_list)
                self.yam_output_written = True
            if PR2_MOTION_ENABLE:
                self.state = self.PR2_STATE_EXECUTE

    def __publish_collision_cloud(self, excluded_object):
        """Publish the necessary collision data to MoveIt!

        Args:
            excluded_object: name of the detected object to exclude from
            publishing its cloud. Usually that's the object to be picked next
        """
        self.__clear_collision_map()
        assert self.ros_cloud_left_table is not None
        self.pcl_left_table_pub.publish(self.ros_cloud_left_table)
        self.collision_points_pub.publish(self.ros_cloud_left_table)
        assert self.ros_cloud_right_table is not None
        self.pcl_right_table_pub.publish(self.ros_cloud_right_table)
        self.collision_points_pub.publish(self.ros_cloud_right_table)
        assert self.perception.ros_cloud_table is not None
        self.collision_points_pub.publish(self.perception.ros_cloud_table)
        rospy.logwarn(
            'publish collision data except {}'.format(excluded_object))
        for do in self.perception.detected_objects_list:
            if do.label != excluded_object:
                rospy.logwarn('publish {} collision data'.format(do.label))
                self.collision_points_pub.publish(do.cloud)

    def __update_execute(self):
        """Update the execute state logic"""

        # In case of a non-empty task list, execute the next task, otherwise
        # transition to done state
        if self.current_task_idx < len(self.task_list):
            task = self.task_list[self.current_task_idx]

            self.__publish_collision_cloud(task.object_name.data)

            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')
            try:
                pick_place_routine = rospy.ServiceProxy(
                    'pick_place_routine', PickPlace)

                resp = pick_place_routine(
                    task.test_scene_num,
                    task.object_name,
                    task.arm_name,
                    task.pick_pose,
                    task.place_pose)

                rospy.loginfo('Response: {}'.format(resp.success))
                if resp.success:
                    rospy.loginfo('{} picked successfully'.format(
                        task.object_name.data))

            except rospy.ServiceException, e:
                rospy.logerr('Service call failed: {}'.format(e))
            self.current_task_idx += 1
        else:
            self.state = self.PR2_STATE_DONE

    def __rotate_at_goal(self, goal_j1):
        """Check if the PR2 is rotated in-place within tolerance.

        Args:
            goal_j1: the goal rotation angle in rad to check against.

        Returns:
            True if the goal is reached, False otherwise.
        """
        joint_state = rospy.wait_for_message(
            '/pr2/joint_states', JointState)
        curr_j1 = joint_state.position[PR2_WORLD_JOINT_IDX]
        return Pr2StateMachine.__rotate_at_goal_helper(curr_j1, goal_j1)

    @staticmethod
    def __rotate_at_goal_helper(pos_j1, goal_j1):
        """Check if 2 angles are close within tolerance.

        Args:
            pos_j1: the first angle in rad.
            goal_j1: the second angle in rad.

        Returns:
            True if the 2 angles are close, False otherwise.
        """
        tolerance = .01
        result = abs(pos_j1 - goal_j1) <= abs(tolerance)
        return result

    def __rotate_in_place(self, pos_j1):
        """Rotate the PR2 in-place.

        The method is a blocking call and won't return unless the rotation
        is completed successfully or an error occurred.

        Args:
            pos_j1: target rotation angle in rad.
        """
        time_elapsed = rospy.Time.now()
        self.joint_controller_pub.publish(pos_j1)
        while True:
            joint_state = rospy.wait_for_message(
                '/pr2/joint_states', JointState)

            curr_j1 = joint_state.position[PR2_WORLD_JOINT_IDX]
            if Pr2StateMachine.__rotate_at_goal_helper(curr_j1, pos_j1):
                time_elapsed = joint_state.header.stamp - time_elapsed
                break

        rospy.loginfo('PR2 world_joint rotated to {} in {}s'.format(
            pos_j1, time_elapsed.to_sec()))


def pcl_callback(pcl_msg):
    """A callback for every received PointCloud2 msg"""
    global frame_num

    frame_num += 1
    rospy.loginfo('### PCL BEGIN Frame{} ###'.format(frame_num))

    pcl_cloud = ros_to_pcl(pcl_msg)
    pr2_perception.update(pcl_cloud)

    # Update the perception pipeline twice faster than the state-machine
    # It was found that updating both at the same rate result in the
    # state-machine operating sometimes on a stale PointCloud msgs
    if frame_num % 2:
        pr2_state_machine.update()

    rospy.loginfo('### PCL END ##')


if __name__ == '__main__':

    rospy.init_node('pr2_pick_place', anonymous=True)

    get_color_list.color_list = []
    frame_num = 0

    pr2_perception = Pr2Perception()
    pr2_state_machine = Pr2StateMachine(pr2_perception)

    pcl_sub = rospy.Subscriber(
        "/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    while not rospy.is_shutdown():
        rospy.spin()
