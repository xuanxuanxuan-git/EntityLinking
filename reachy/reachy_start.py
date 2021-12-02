# Adopted from codes by Kirill Kokorin
# Modified by Yueqing Xuan

# This file contains codes that are related to the executions of tasks by reachy.

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode
import numpy as np
from math import sin, cos, pi
import time
import sys


# Connect to the robot via IP (wifi or ethernet)
def connect_to_robot(robot_ip):
    # If this times out you may need to connect the screen, keyboard
    # and mouse to the robot and check the network settings
    robot_obj = ReachySDK(robot_ip)
    print('Connected to IP %s' % robot_ip)
    return robot_obj


# Turn on the motors, the arm will become stiff
def turn_on_motors(reachy):
    reachy.turn_on('reachy')
    reachy.turn_on('r_arm')
    print('Turned reachy and right arm')


# Turn motors off, the arm will fall so you should be hovering very close
# to the table
def turn_off_motors(reachy):
    reachy.turn_off_smoothly('r_arm')
    print('Turned off right arm')


# Print joint angles
def print_joint_angles(reachy):
    Q = [j.present_position for j in reachy.r_arm.joints.values()]
    joints = Q[0:-1]
    gripper = Q[-1]
    print('--->Joint angles<---')
    print('Shoulder (pitch, roll, yaw): (%.1f, %.1f, %.1f)' % (Q[0], Q[1], Q[2]))
    print('Elbow (pitch, yaw): (%.1f, %.1f)' % (Q[3], Q[4]))
    print('Wrist (pitch, roll): (%.1f, %.1f)' % (Q[5], Q[6]))
    print('Gripper: %.1f' % gripper)
    return joints, gripper


# Print motor temperatures, fan will turn on at 45C and the motor
# turns off at 55C
def print_motor_temp(reachy):
    temps = [
        reachy.r_arm.r_shoulder_pitch.temperature,
        reachy.r_arm.r_shoulder_roll.temperature,
        reachy.r_arm.r_arm_yaw.temperature,
        reachy.r_arm.r_elbow_pitch.temperature,
        reachy.r_arm.r_forearm_yaw.temperature,
        reachy.r_arm.r_wrist_pitch.temperature,
        reachy.r_arm.r_wrist_roll.temperature,
        reachy.r_arm.r_gripper.temperature,
    ]
    print('***Motor temperatures (45=fan, 55=shutdown)***')
    print('Shoulder (pitch, roll, yaw): (%.1f, %.1f, %.1f)' % (temps[0], temps[1], temps[2]))
    print('Elbow (pitch, yaw): (%.1f, %.1f)' % (temps[3], temps[4]))
    print('Wrist (pitch, roll): (%.1f, %.1f)' % (temps[5], temps[6]))
    print('Gripper: %.1f' % temps[7])


# Print pose matrix
def print_pose(pose):
    print('Rotation matrix')
    print('[%.3f, %.3f, %.3f]' % (pose[0, 0], pose[0, 1], pose[0, 2]))
    print('[%.3f, %.3f, %.3f]' % (pose[1, 0], pose[1, 1], pose[2, 2]))
    print('[%.3f, %.3f, %.3f]' % (pose[2, 0], pose[2, 1], pose[2, 2]))
    print('Translation: (%.3f, %.3f, %.3f)' % (pose[0, 3], pose[1, 3], pose[2, 3]))


# Get rotation matrix based on axis and angle in deg
def rot_mat(q, direction):
    q = q / 180 * pi
    if direction == 'x':
        R = np.array([[1, 0, 0],
                      [0, cos(q), -sin(q)],
                      [0, sin(q), cos(q)]])
    elif direction == 'y':
        R = np.array([[cos(q), 0, sin(q)],
                      [0, 1, 0],
                      [-sin(q), 0, cos(q)]])
    elif direction == 'z':
        R = np.array([[cos(q), -sin(q), 0],
                      [sin(q), cos(q), 0],
                      [0, 0, 1]])
    else:
        R = np.eye(3)
    return R


# Move the arm home
def go_home(reachy, home_joint_angles, move_duration):
    if reachy.r_arm.r_shoulder_pitch.compliant == False:
        home_position = {
            reachy.r_arm.r_shoulder_pitch: home_joint_angles[0],
            reachy.r_arm.r_shoulder_roll: home_joint_angles[1],
            reachy.r_arm.r_arm_yaw: home_joint_angles[2],
            reachy.r_arm.r_elbow_pitch: home_joint_angles[3],
            reachy.r_arm.r_forearm_yaw: home_joint_angles[4],
            reachy.r_arm.r_wrist_pitch: home_joint_angles[5],
            reachy.r_arm.r_wrist_roll: home_joint_angles[6],
            reachy.r_arm.r_gripper: home_joint_angles[7]
        }
        goto(
            goal_positions=home_position,
            duration=move_duration,
            interpolation_mode=InterpolationMode.MINIMUM_JERK
        )
        print('Completed moving home')
    else:
        print('Arm motors are turned off')


# Move the gripper to location specified by coordinates tuple (x,y,z) in m,
# must be in the frame of reference of the robot
# hold=True will keep the arm in this location, but motors will give out
# open_gripper=True will open the jaws so an object can be grasped
def go_to_location(reachy, coords, theta_gripper, open_gripper, move_duration):
    GRIPPER_FULLY_OPEN = -30
    FORWARD_TILT_COMPENSATE = 100

    # Check that arm is on
    if reachy.r_arm.r_shoulder_pitch.compliant == False:
        init_joints, init_gripper = print_joint_angles(reachy)
        init_pose = reachy.r_arm.forward_kinematics(init_joints)
        print('Starting position')
        print_pose(init_pose)

        # Turn gripper towards target
        goal_rot = np.matmul(rot_mat(theta_gripper, 'z'),
                             rot_mat(-FORWARD_TILT_COMPENSATE, 'y'))
        goal_pose = np.zeros((4, 4))
        goal_pose[0:3, 0:3] = goal_rot
        goal_pose[0:4, 3] = np.array([[coords[0], coords[1], coords[2], 1]])
        print('Goal position')
        print_pose(goal_pose)

        # Get joint angles and move to pose
        goal_joint_angles = reachy.r_arm.inverse_kinematics(goal_pose, init_joints)
        goto({
            reachy.r_arm.r_shoulder_pitch: goal_joint_angles[0],
            reachy.r_arm.r_shoulder_roll: goal_joint_angles[1],
            reachy.r_arm.r_arm_yaw: goal_joint_angles[2],
            reachy.r_arm.r_elbow_pitch: goal_joint_angles[3],
            reachy.r_arm.r_forearm_yaw: goal_joint_angles[4],
            reachy.r_arm.r_wrist_pitch: goal_joint_angles[5],
            reachy.r_arm.r_wrist_roll: goal_joint_angles[6],
            reachy.r_arm.r_gripper: GRIPPER_FULLY_OPEN if open_gripper else init_gripper
        }, duration=move_duration)

    else:
        print('Arm motors are turned off')


# Pickup object at (x,y,z)
def pick_up(reachy, coords, gripper_angle, move_duration):
    PICKUP_FORCE = 10
    GRIPPER_FULLY_CLOSED = 22
    GRIPPER_CLOSE_SPEED = 1.0
    GRIP_STEP_ANGLE = 5
    SET_HOVER_HEIGHT = 0.1  # m
    RAISE_MOVE_SPEED = 2

    hover_height = SET_HOVER_HEIGHT
    gripper_pos = reachy.r_arm.r_gripper.present_position
    go_to_location(reachy, coords, gripper_angle, open_gripper=True, move_duration=move_duration)

    # while reachy.force_sensors.r_force_gripper.force < PICKUP_FORCE:
    #     gripper_pos = reachy.r_arm.r_gripper.present_position
    #     if gripper_pos > GRIPPER_FULLY_CLOSED:
    #         print('Nothing grasped')
    #         return False, hover_height #Closed without object
    #     goto({reachy.r_arm.r_gripper: gripper_pos + GRIP_STEP_ANGLE}, duration=GRIPPER_CLOSE_SPEED)
    #     print('Finished grasp')

    goto({reachy.r_arm.r_gripper: GRIPPER_FULLY_CLOSED},
         duration=GRIPPER_CLOSE_SPEED)
    print('Finished grasp')

    # Move slightly up to hover the object above the table
    raised_coords = (coords[0], coords[1], coords[2] + SET_HOVER_HEIGHT)
    go_to_location(reachy, raised_coords, gripper_angle, open_gripper=False,
                   move_duration=RAISE_MOVE_SPEED)

    return True, hover_height  # Grasp succeeded (reached force limit)


# Place object
def place(reachy, coords, gripper_angle, move_duration):
    GRIPPER_OPEN_SPEED = 1.0
    SET_DROP_HEIGHT = 0.1  # m

    #     #Check for object
    #     if reachy_robot.force_sensors.r_force_gripper.force < HOLDING_FORCE:
    #         return False #No object held
    #     else:
    # Move in xy plane
    joints, gripper = print_joint_angles(reachy)
    init_pose = reachy.r_arm.forward_kinematics(joints)
    #     coords = (coords[0], coords[1], coords[2]+SET_DROP_HEIGHT)
    go_to_location(reachy, coords, gripper_angle, open_gripper=False,
                   move_duration=move_duration)

    # Move down
    dropped_coords = (coords[0], coords[1], coords[2] - SET_DROP_HEIGHT)
    go_to_location(reachy, dropped_coords, gripper_angle,
                   open_gripper=False, move_duration=GRIPPER_OPEN_SPEED)

    # Drop the object
    go_to_location(reachy, dropped_coords, gripper_angle,
                   open_gripper=True, move_duration=GRIPPER_OPEN_SPEED)

    return True


# Print the current arm position
def print_current_pose(reachy):
    init_joints, init_gripper = print_joint_angles(reachy)
    init_pose = reachy.r_arm.forward_kinematics(init_joints)
    print_pose(init_pose)


# run this every couple of hours to prevent the robot from doing weird actions
# before running this function, make the arm facing toward the bottom
def reset_coordinate(reachy):
    Q = [0, 0, 0, 0, 0, 0, 0, 0]
    turn_on_motors(reachy)
    go_home(reachy, Q, 4)
    # print(reachy.r_arm.forward_kinematics())
    reachy.turn_off_smoothly('r_arm')


# place the arm at a position and get the information of the pose
def get_current_position(reachy):
    turn_on_motors(reachy)
    print_current_pose(reachy)
    reachy.turn_off_smoothly('r_arm')


# Parameters
ROBOT_IP = '192.168.1.1'  # may need to change this
HOME_POSITION = [30, 0, 0, -120, 0, -15, 0, 15]  # gripper aligned to chest
RIGHT_ANGLED_POSITION = [10, 0, 0, -100, 0, -5, 0, 15]
SLOW_MOVE_SPEED = 3
FAST_MOVE_SPEED = 1.5

# Positions
# POS1 = (0.37, -0.20, -0.20)
POS1 = (0.45, -0.20, -0.20)
POS1_ANGLE = 0
POS2 = (0.34, -0.34, -0.20)
POS2_ANGLE = -20

# Arm will fall after reaching point 3 and point 6
# Please refer to onenote (RA week 3 -> demos) for more information
POS3 = (0.55, -0.18, -0.20)
POS3_ANGLE = 0

POS4 = (0.30, -0.06, -0.20)
POS4_ANGLE = 0
POS5 = (0.42, 0.00, -0.20)
POS5_ANGLE = 0
POS_6 = (0.56, -0.11, -0.1)
POS6_ANGLE = 5
# ----------------------- Execution ------------------------------- #


# Actions: pick up the target and place it on top of a receptacle
def go_pick_go_place(reachy, pick_point, place_point):
    turn_on_motors(reachy)
    pick_coord, pick_point_angle = pick_point
    place_coord, place_point_angle = place_point

    go_home(reachy, HOME_POSITION, FAST_MOVE_SPEED)
    status, lift_height = pick_up(reachy, pick_coord, pick_point_angle, SLOW_MOVE_SPEED)

    if status:
        place_coord = (place_coord[0], place_coord[1], place_coord[2] + lift_height)

    place(reachy, place_coord, place_point_angle, SLOW_MOVE_SPEED)

    go_home(reachy, HOME_POSITION, SLOW_MOVE_SPEED)
    time.sleep(2)
    go_home(reachy, RIGHT_ANGLED_POSITION, SLOW_MOVE_SPEED)
    turn_off_motors(reachy)
    print("task completed!")
    return True


# Actions: go to the target and pick it up. Then hold the object for 3 seconds and go
# home at the end.
def go_to(reachy, point, pick_action=True):
    turn_on_motors(reachy)
    go_home(reachy, HOME_POSITION, FAST_MOVE_SPEED)
    pos_coord, pos_angle = point
    if not pick_action:
        go_to_location(reachy, pos_coord, pos_angle,
                       open_gripper=False, move_duration=SLOW_MOVE_SPEED)
    else:
        pick_up(reachy, pos_coord, pos_angle, SLOW_MOVE_SPEED)

    time.sleep(3)
    go_home(reachy, HOME_POSITION, FAST_MOVE_SPEED)
    time.sleep(2)
    go_home(reachy, RIGHT_ANGLED_POSITION, FAST_MOVE_SPEED)
    turn_off_motors(reachy)
    print("task completed")
    return True


# Actions: turn on the motor, then initialise the arm but do nothing
def start_and_rest(reachy):
    turn_on_motors(reachy)
    go_home(reachy, HOME_POSITION, FAST_MOVE_SPEED)
    time.sleep(2)
    go_home(reachy, RIGHT_ANGLED_POSITION, FAST_MOVE_SPEED)
    turn_off_motors(reachy)
    return True


# run the program to reset the reachy's coordinate
if __name__ == "__main__":
    # Connecting to the robot (only do this once per session)
    reachy_robot = connect_to_robot(ROBOT_IP)

    # reset reachy's coordinate
    reset_coordinate(reachy_robot)

    # get current position of reachy
    get_current_position(reachy_robot)

