"""student_controller controller."""

import math
import numpy as np


class StudentController:
    def __init__(self):
        # init starting robot pose
        self.poses = []
        self.current_pose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.poses.append(self.current_pose)

        # landmark storage
        self.landmarks = {}
        
        # accumulated ds and dtheta between keyframes
        self.accumulated_ds = 0.0
        self.accumulated_dtheta = 0.0

        # graph factors
        self.prior_factors = []
        self.odom_factors = []
        self.obs_factors = []

        # key frame and optimization params
        self.translation_threshold = 0.05
        self.rotation_threshold = 0.1
        self.optimize_every = 5
        
    # HELPER METHODS (there a lot of them):

    # angle wrapper helper
    def wrap_angle(self, theta):
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    # propagate pose through diff drive motion model (may want to replace with EKF later)
    def propagate_pose(self, pose: np.ndarray, ds: float, dtheta: float) -> np.ndarray:
        x, y, theta = pose

        x_new = x + ds * np.cos(theta + 0.5 * dtheta)
        y_new = y + ds * np.sin(theta + 0.5 * dtheta)
        theta_new = self.wrap_angle(theta + dtheta)

        return np.array([x_new, y_new, theta_new], dtype=float)
    
    def keyframe_decision(self, last_pose: np.ndarray, current_pose: np.ndarray, ds_thresh: float, dtheta_thresh: float):
        dx = current_pose[0] - last_pose[0]
        dy = current_pose[1] - last_pose[1]

        dtheta = self.wrap_angle(current_pose[2] - last_pose[2])
        ds = np.hypot(dx, dy)

        return (ds > ds_thresh) or (dtheta > dtheta_thresh)
    
    # TODO: may need to investigate bearing phi
    def intit_landmark(self, landmark_id, pose: np.ndarray, r: float, phi: float):
        x, y, theta = pose

        mx = x + r * np.cos(theta + phi)
        my = y + r * np.sin(theta + phi)

        return np.array([mx, my], dtype=float)
    
    def compute_prior_factor(self, pose: np.ndarray, measurement: np.ndarray):
        x, y, theta = pose
        x_meas, y_meas, theta_meas = measurement

        e = np.array([[x - x_meas], 
                      [y - y_meas], 
                      [self.wrap_angle(theta - theta_meas)]], dtype=float)
        J = np.identity(3)

        return e, J
    
    def compute_odom_factor(self, pose_i: np.ndarray, pose_j: np.ndarray, measurement: np.ndarray):
        x_i, y_i, theta_i = pose_i
        x_j, y_j, theta_j = pose_j
        
        ds_meas = measurement[0]
        # measurement[1] is 0.0
        dtheta_meas = measurement[2]

        c = np.cos(theta_i)
        s = np.sin(theta_i)
        R = np.array([[c, s], 
                      [-s, c]])

        dx = x_j - x_i
        dy = y_j - y_i
        dtheta = theta_j - theta_i
        d = np.array([dx, dy])

        d_rot = R @ d
        e = np.array([[d_rot[0] - ds_meas], 
                      [d_rot[1] - 0.0],
                      [self.wrap_angle(dtheta - dtheta_meas)]], dtype=float)

        A = np.array([[-c, -s, -s*dx + c*dy],
                      [s, -c, -c*dx - s*dy],
                      [0, 0, -1]])
        B = np.array([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])
        
        return e, A, B
    
    def compute_obs_factor(self, pose: np.ndarray, landmark: np.ndarray, measurement: np.ndarray):
        """
        measurement: np.array([r, phi])
        Returns e (2x1 array), Jx (2x3 array), Jm (2x2 array)
        """
        x, y, theta = pose
        mx, my = landmark
        r_meas, phi_meas = measurement

        dx = mx - x
        dy = my - y

        r_pred = np.hypot(dx, dy)
        phi_pred = self.wrap_angle(np.arctan2(dy, dx) - theta)

        e = np.array([[r_pred - r_meas],
                      [self.wrap_angle(phi_pred - phi_meas)]], dtype=float)

        Jx = np.array([[-dx/r_pred, -dy/r_pred, 0],
                      [dy/(r_pred**2), -dx/(r_pred**2), -1]])
        Jm = np.array([[dx/r_pred, dy/r_pred],
                      [-dy/(r_pred**2), dx/(r_pred**2)]])

        return e, Jx, Jm
    
    # TODO IMPLEMENT LATER
    def run_graph_optimization(self):
        num_poses = len(self.poses)
        num_landmarks = len(self.landmarks)

        # total size of state vector
        dim = (num_poses * 3) + (num_landmarks * 2)
        
        pass
    
    # LOOP:
    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict:   dict, contains control for "left_motor" and "right_motor"
        estimated_pose: list, contains float values representing the robot's pose,
                        (x,y,orientation).
                        The pose should be given using a right-handed coordinate
                        system: positive x is the right side of the arena, positive
                        y is the top side of the arena, theta increases as the
                        robot turns counter-clockwise.
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # TODO: add your controllers here.
        control_dict["left_motor"] = 3.0
        control_dict["right_motor"] = 3.0

        # TODO: change later
        estimated_pose = [0, 0, 0]
        estimated_map = {"BOX_1": [0.5, 0.5]}

        # ds and dtheta from odometry
        ds = sensors["odometry"][0]
        dtheta = sensors["odometry"][1]

        # accumulated ds and dtheta
        self.accumulated_ds += ds
        self.accumulated_dtheta += dtheta
        self.accumulated_dtheta = self.wrap_angle(self.accumulated_dtheta)

        # update current_pose using motion model
        self.current_pose = self.propagate_pose(self.current_pose, ds, dtheta)

        # when new keyframe threshold reached
        if self.keyframe_decision(self.poses[-1], self.current_pose, self.translation_threshold, self.rotation_threshold):
            # calculate indices for last and new keyframe poses and add a copy of the current pose to poses
            prev_pose_idx = len(self.poses) - 1
            self.poses.append(self.current_pose.copy())
            current_pose_idx = len(self.poses) - 1

            # build odom factor dictionary to track last keyframe pose and new keyframe pose and difference between
            new_odom_factor = {}
            new_odom_factor["i"] = prev_pose_idx
            new_odom_factor["j"] = current_pose_idx
            new_odom_factor["measurement"] = np.array([self.accumulated_ds, 0.0, self.accumulated_dtheta], dtype=float)

            # add new odom factor to global list
            self.odom_factors.append(new_odom_factor)

            # reset accumulation for next keyframe
            self.accumulated_ds = 0.0
            self.accumulated_dtheta = 0.0

            # scan for landmarks
            for landmark_id, (r_meas, phi_meas) in sensors["observed_landmarks"].items():
                if landmark_id not in self.landmarks:
                    # add landmark to global dictionary
                    self.landmarks[landmark_id] = self.intit_landmark(landmark_id, self.current_pose, r_meas, phi_meas)

                # build obs factor dictionary to track current observations and measurement
                new_obs_factor = {}
                new_obs_factor["pose_idx"] = current_pose_idx
                new_obs_factor["landmark_id"] = landmark_id
                new_obs_factor["measurement"] = np.array([r_meas, phi_meas], dtype=float)
                self.obs_factors.append(new_obs_factor)

            if len(self.poses) % self.optimize_every == 0:
                self.run_graph_optimization()


        estimated_pose = self.current_pose
        estimated_map = self.landmarks

        return control_dict, estimated_pose, estimated_map
