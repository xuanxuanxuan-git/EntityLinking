# Yueqing Xuan
# 1075355

# This file contains the codes that will transform the coordinate system of the camera
# to the coordinate system of the robot. The transformation contains rotation and translation.
# The the transformation matrix will apply to the coordinates of the objects that are detected
# by the camera. The results will be coordinates of the objects from reachy's perspective.

from scipy.spatial.transform import Rotation as R
import numpy as np
import math

# Four positions of the camera (relative to the position of the robot)
OPPOSITE = "opposite"
SAME = "same"
LEFT = "left"
RIGHT = "right"


# define all the rotation matrices
# theta is the angle of the camera (the angle between the camera and the ground)
def define_all_matrices(theta):
    radian = math.radians(theta)
    sin_angle = math.sin(radian)
    cos_angle = math.cos(radian)

    # rotate along the x-axis by "theta" degrees (anti-clockwise)
    R_x1 = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle],
    ])

    # rotate along x-axis by +90 degrees (anti-clockwise)
    R_x2 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])

    # rotate along z-axis by -90 degrees (clockwise)
    R_z1 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ])

    # rotate along z-axis by +90 degrees (anti-clockwise)
    R_z2 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    # rotate along z-axis by 180 degrees
    R_z3 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ])

    return R_x1, R_x2, R_z1, R_z2, R_z3


# Calculate the complete rotation matrix based on the position of the camera and the robot
def calculate_rotation_matrix(direction, theta):
    M = None
    R_x1, R_x2, R_z1, R_z2, R_z3 = define_all_matrices(theta)

    if direction == OPPOSITE:  # camera and robot is facing to each other
        M_opposite = R_x1.dot(R_x2).dot(R_z1)
        M = M_opposite
    elif direction == SAME:  # camera and robot is facing to the same direction
        M_same = R_x1.dot(R_x2).dot(R_z2)
        M = M_same
    elif direction == LEFT:  # camera is on the left hand side of the robot
        M_left = R_x1.dot(R_x2).dot(R_z3)
        M = M_left
    elif direction == RIGHT:  # camera is on the right hand side of the robot
        M_right = R_x1.dot(R_x2)
        M = M_right

    return M


# use scipy to understand what this matrix represents.
def print_matrices(m):
    print(m)
    print("orientation angles: {}\n".format(R.from_matrix(m).as_euler('xyz', degrees=True)))


# Limiting floats to two decimal points
def two_decimal(x):
    return round(x, 2)


# transform coordinates on a single vector
# Since the height estimation from the camera is not accurate, we just set the height to
# a fixed value, an in this case, -0.2m (from the robot's perdpective)
def rotate_and_translate(coord, translation_vector, rotation_matrix):
    x, y, z = coord
    d_x, d_y, d_z = translation_vector
    obj_vector = np.array([x, y, z])
    obj_vector_r = obj_vector.dot(rotation_matrix)
    obj_vector_t = (two_decimal(obj_vector_r[0] + d_x),
                    two_decimal(obj_vector_r[1] + d_y),
                    # two_decimal(obj_vector_r[2] + d_z))
                    -0.2)
    # print(obj_vector)
    # print(obj_vector_r)
    # print(obj_vector_t)

    return obj_vector_t


# transform the coordinates in the camera's system to the coordinates in the robot's system
# this transformation applies to coordinates of all objects detected by the camera
def coordinate_transformation(entity_list, direction, theta, translation_vector):
    new_list = []
    M = calculate_rotation_matrix(direction, theta)

    for item in entity_list:
        new_item = item
        new_item["position"] = rotate_and_translate(item["position"], translation_vector, M)
        new_list.append(new_item)

    return new_list


if __name__ == '__main__':
    theta = 30
    matrices = define_all_matrices(theta)
    for m in matrices:
        print_matrices(m)
