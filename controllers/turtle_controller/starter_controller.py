"""student_controller controller."""

import math
import numpy as np
import matplotlib.pyplot as plt

# NOTE: Sometimes my random walk gets really unlucky and I end up missing one of the boxes in the first 5 minutes.
#       It's not common but it's theoretically possible so if it does please rerun it!
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

        # TODO: tune these
        # key frame and optimization params 
        self.translation_threshold = 0.25
        self.rotation_threshold = 0.3
        self.optimize_every = 10

        # ekf for a fast updates
        self.mu = np.array([0.0, 0.0, 0.0])
        self.Sigma = np.diag([0.001, 0.001, 0.001])

        # for fsm drive behavior
        self.robot_state = "EXPLORE"
        self.turn_target = 0.0

        # for trajectory plotting
        self.init_plot()
        self.step_count = 0

        # box counter for unknown correspondence
        self.box_counter = 0


    def init_plot(self):
        """Sets up the live matplotlib window."""
        plt.ion()  # Turn on interactive mode so it doesn't block Webots
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        # Based on your 5x5m arena image, -3 to 3 should cover everything perfectly
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_title("Live GraphSLAM Map")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Initialize empty plot objects that we will update rapidly
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=2, label='Estimated Path')
        self.landmark_scatter = self.ax.scatter([], [], c='red', marker='s', s=100, label='Boxes')
        self.robot_arrow = None  # Placeholder for the robot's heading arrow
        
        self.ax.legend(loc='upper right')
        plt.show()

    def update_plot(self):
        """Rapidly refreshes the plot with current graph state."""
        # 1. Update the robot path (blue line)
        if len(self.poses) > 0:
            xs = [p[0] for p in self.poses]
            ys = [p[1] for p in self.poses]
            self.path_line.set_data(xs, ys)
            
            # 2. Update the live robot position and heading (green arrow)
            if self.robot_arrow:
                self.robot_arrow.remove()
            
            current_x, current_y, current_theta = self.current_pose
            # Calculate arrow direction
            dx = 0.2 * np.cos(current_theta)
            dy = 0.2 * np.sin(current_theta)
            self.robot_arrow = self.ax.arrow(current_x, current_y, dx, dy, 
                                             head_width=0.1, head_length=0.1, 
                                             fc='green', ec='green')

        # 3. Update the landmarks (red squares)
        if len(self.landmarks) > 0:
            lx = [m[0] for m in self.landmarks.values()]
            ly = [m[1] for m in self.landmarks.values()]
            self.landmark_scatter.set_offsets(np.c_[lx, ly])

        # Force matplotlib to draw the new frame immediately
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()    
        
    def ekf_predict(self, ds: float, dtheta: float):
        x_prev = self.mu[0]
        y_prev = self.mu[1]
        theta_prev = self.mu[2]

        theta = theta_prev + dtheta

        # jacobian of a differential drive motion model (called A in the math i read)
        A = np.array([[1, 0, -ds*np.sin(theta_prev)],
                        [0, 1, ds*np.cos(theta_prev)], 
                        [0, 0, 1]
        ])

        # mean state as predicted by motion model
        mu_bar = np.array([x_prev + ds*np.cos(theta_prev), 
                        y_prev + ds*np.sin(theta_prev),
                        np.arctan2(np.sin(theta), np.cos(theta))
        ])

        # simple noise, tune with velocity and coefficients later
        Q = np.diag([ 
            (0.1 * ds + 0.005)**2, 
            (0.1 * ds + 0.005)**2, 
            (0.1 * abs(dtheta) + 0.005)**2 
        ])

        # compute covariance matrix
        Sigma_bar = A @ self.Sigma @ A.T + Q

        # update our global sigma and mu
        self.mu = mu_bar
        self.Sigma = Sigma_bar

    # this method is not being used right now due to lidar noise but i'm leaving it in for reference
    def ekf_measurement(self, x_j: float, y_j: float, r_meas: float, phi_meas: float):
        z = np.array([r_meas, phi_meas])

        dx = x_j - self.mu[0]
        dy = y_j - self.mu[1]

        q = dx**2 + dy**2

        r_pred = np.sqrt(q)
        phi_pred = np.arctan2(dy, dx) - self.mu[2]
        phi_pred = np.arctan2(np.sin(phi_pred), np.cos(phi_pred))

        z_hat = np.array([r_pred, phi_pred])

        # observation model jacobian
        H = np.array([[-dx/r_pred, -dy/r_pred, 0],
                      [dy/q, -dx/q, -1]
        ])

        # no sensor noise in compass esentially
        R = np.diag([0.15**2, 0.05**2])

        # innovation covariance
        S = H @ self.Sigma @ H.T + R

        # kalman gain
        K = self.Sigma @ H.T @ np.linalg.inv(S) 

        # measurement error also wrapping angle
        y = z - z_hat
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        return H, S, K, y
    
    def ekf_update_compass(self, heading_meas: float):
        # trust the compass because noise is 0
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[0.000001]])
        
        # innovation angle
        y = np.array([self.wrap_angle(heading_meas - self.mu[2])])
        
        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        # ppdate state and covariance and flatten
        self.mu = self.mu + (K @ y).flatten()
        self.mu[2] = self.wrap_angle(self.mu[2])
        self.Sigma = (np.identity(3) - K @ H) @ self.Sigma

    # GRAPH HELPER METHODS (there a lot of them):
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
    def init_landmark(self, pose: np.ndarray, r: float, phi: float):
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
        dy_meas = measurement[1]
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
                      [d_rot[1] - dy_meas],
                      [self.wrap_angle(dtheta - dtheta_meas)]], dtype=float)

        A = np.array([[-c, -s, -s*dx + c*dy],
                      [s, -c, -c*dx - s*dy],
                      [0, 0, -1]])
        B = np.array([[c, s, 0],
                      [-s, c, 0],
                      [0, 0, 1]])
        
        return e, A, B
    
    def compute_obs_factor(self, pose: np.ndarray, landmark: np.ndarray, measurement: np.ndarray):
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
    
    # hopefully my computer dont break :(
    def run_graph_optimization(self):
        num_poses = len(self.poses)
        num_landmarks = len(self.landmarks)

        # total size of state vector
        dim = (num_poses * 3) + (num_landmarks * 2)

        # TODO: Tune this
        num_iterations = 8 
        for iteration in range(num_iterations):
            # size hessian
            H = np.zeros((dim,dim), dtype=float)
            g = np.zeros((dim, 1), dtype=float)

            # landmark keys
            landmark_list = list(self.landmarks.keys())
            # map key indexes to keys
            landmark_idx_map = {l_id: i for i, l_id in enumerate(landmark_list)}

            # TODO: information matrix; tune based on turtle_controller noise
            omega_prior = np.identity(3) * 1000000.0
            omega_odom = np.diag([40000.0, 10000.0, 40000.0])
            omega_obs = np.diag([44.4, 100.0])

            # compute residual function and jacobian for prior
            e_prior, J_prior = self.compute_prior_factor(self.poses[0], np.array([0.0, 0.0, 0.0]))

            # build hessian and gradient for optimization problem using what i computed in prior
            H_prior = J_prior.T @ omega_prior @ J_prior
            g_prior = J_prior.T @ omega_prior @ e_prior

            # add to global hessian and gradient
            H[0:3, 0:3] += H_prior
            g[0:3, 0:] += g_prior

            # loop through odom factors
            for factor in self.odom_factors:
                i = factor["i"]
                j = factor["j"]
                meas = factor["measurement"]

                # compute residual function and jacobian(s) for prior
                e_odom, A, B = self.compute_odom_factor(self.poses[i], self.poses[j], meas)

                # more local hessians and gradients but this time for odometry factors
                H_ii = A.T @ omega_odom @ A
                H_ij = A.T @ omega_odom @ B
                H_ji = B.T @ omega_odom @ A
                H_jj = B.T @ omega_odom @ B
                g_i = A.T @ omega_odom @ e_odom
                g_j = B.T @ omega_odom @ e_odom

                # continue to build global hessians in corresponding indices
                H[i*3 : i*3+3, i*3 : i*3+3] += H_ii
                H[i*3 : i*3+3, j*3 : j*3+3] += H_ij
                H[j*3 : j*3+3, i*3 : i*3+3] += H_ji
                H[j*3 : j*3+3, j*3 : j*3+3] += H_jj
                
                g[i*3 : i*3+3, 0:] += g_i
                g[j*3 : j*3+3, 0:] += g_j

            # loop through obs_factors
            for factor in self.obs_factors:
                i = factor["pose_idx"]
                l_id = factor["landmark_id"]
                meas = factor["measurement"]
                
                # final residual func and jacobian for obs
                e_obs, Jx, Jm = self.compute_obs_factor(self.poses[i], self.landmarks[l_id], meas)
                
                # more local hessians
                H_xx = Jx.T @ omega_obs @ Jx
                H_xm = Jx.T @ omega_obs @ Jm
                H_mx = Jm.T @ omega_obs @ Jx
                H_mm = Jm.T @ omega_obs @ Jm
                g_x = Jx.T @ omega_obs @ e_obs
                g_m = Jm.T @ omega_obs @ e_obs

                # TODO: make this global to method for legibility sake (i reuse this variable like 3 times)
                # start and end index for pose
                p_start = i * 3
                p_end = p_start + 3

                # start and end index for map coords
                m_start = (num_poses * 3) + (landmark_idx_map[l_id] * 2)
                m_end = m_start + 2

                # adding local hessians to global hessian based on coords
                H[p_start:p_end, p_start:p_end] += H_xx  # pose vs pose
                H[p_start:p_end, m_start:m_end] += H_xm  # pose vs landmark
                H[m_start:m_end, p_start:p_end] += H_mx  # landmark vs pose
                H[m_start:m_end, m_start:m_end] += H_mm  # landmark vs landmark

                g[p_start:p_end, 0:] += g_x
                g[m_start:m_end, 0:] += g_m

            # tikhonov regularization (stops landmark pose from being calculate on top of robot causing divide by inf error)
            H += np.eye(dim) * 1e-5

            # computer go brrr (solving optimzation)
            delta = np.linalg.solve(H, -g)

            for i in range(num_poses):
                p_start = i * 3
                p_end = p_start + 3

                # flatten delta to 1d
                pose_delta = delta[p_start:p_end].flatten()

                # apply correction factor to pose
                self.poses[i] += pose_delta
                self.poses[i][2] = self.wrap_angle(self.poses[i][2])

            for l_id, idx in landmark_idx_map.items():
                m_start = (num_poses * 3) + (idx * 2)
                m_end = m_start + 2
                
                # extract the 2x1 correction and flatten it to 1d
                land_delta = delta[m_start:m_end].flatten()
                
                self.landmarks[l_id] += land_delta

        # update current pose with optimized value
        self.current_pose = self.poses[-1].copy()

        self.mu = self.current_pose.copy()
        self.Sigma = np.diag([0.001, 0.001, 0.001])
        
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

        # ekf prediction step (should update global mean covariance)
        self.ekf_predict(ds, dtheta)

        # use compass to update rotation before looking landmarks for ekf
        heading_meas = sensors["heading"]
        self.ekf_update_compass(heading_meas)

        # NOTE: EKF MEASUREMENT CAUSES DRFIT FROM LIDAR NOISE SO I'VE REMOVED IT FOR NOW
        # ekf measurment step (run for each landmark)
        # for landmark_id, (r_meas, phi_meas) in sensors["observed_landmarks"].items():
        #     if landmark_id in self.landmarks:
        #         # test for correspondence
        #         x_j = self.landmarks[landmark_id][0]
        #         y_j = self.landmarks[landmark_id][1]

        #         measurement_data = self.ekf_measurement(x_j, y_j, r_meas, phi_meas)
        #         H = measurement_data[0]
        #         K = measurement_data[2]
        #         y = measurement_data[3]

        #         self.mu = self.mu + K @ y
        #         self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))
        #         self.Sigma = (np.identity(3) - (K @ H)) @ self.Sigma

        # update current_pose using motion model
        self.current_pose = self.mu

        # when new keyframe threshold reached
        if self.keyframe_decision(self.poses[-1], self.current_pose, self.translation_threshold, self.rotation_threshold):
            last_pose = self.poses[-1]
            current_pose = self.mu.copy()
            
            dx = current_pose[0] - last_pose[0]
            dy = current_pose[1] - last_pose[1]
            dtheta = self.wrap_angle(current_pose[2] - last_pose[2])
            
            # Transform global dx, dy into the local frame of the last keyframe
            c = np.cos(last_pose[2])
            s = np.sin(last_pose[2])
            local_dx = c * dx + s * dy
            local_dy = -s * dx + c * dy

            # 2. Add to poses array
            prev_pose_idx = len(self.poses) - 1
            self.poses.append(current_pose)
            current_pose_idx = len(self.poses) - 1

            new_odom_factor = {
                "i": prev_pose_idx,
                "j": current_pose_idx,
                "measurement": np.array([local_dx, local_dy, dtheta], dtype=float)
            }
            self.odom_factors.append(new_odom_factor)

            # scan for landmarks
            for landmark_id, (r_meas, phi_meas) in sensors["observed_landmarks"].items():
                
                matched_landmark = False
                if "BOX_" not in str(landmark_id):    # correspondence is unknown if this is true
                    mx, my = self.init_landmark(self.current_pose, r_meas, phi_meas)

                    for l_id in self.landmarks:
                        mx_comp, my_comp = self.landmarks[l_id]

                        # match if within a reasonable distance of existing box
                        if (abs(mx - mx_comp) < 0.75 and abs(my - my_comp) < 0.75):
                            landmark_id = l_id
                            matched_landmark = True
                            break

                    # create box_id as usual otherwise using box counter
                    if not matched_landmark:
                        landmark_id = f"BOX_{self.box_counter}"
                        self.box_counter += 1

                if landmark_id not in self.landmarks:
                    # add landmark to global dictionary
                    self.landmarks[landmark_id] = self.init_landmark(self.current_pose, r_meas, phi_meas)

                # build obs factor dictionary to track current observations and measurement
                new_obs_factor = {}
                new_obs_factor["pose_idx"] = current_pose_idx
                new_obs_factor["landmark_id"] = landmark_id
                new_obs_factor["measurement"] = np.array([r_meas, phi_meas], dtype=float)
                self.obs_factors.append(new_obs_factor)

            if len(self.poses) % self.optimize_every == 0:
                self.run_graph_optimization()


        estimated_pose = self.current_pose.tolist()
        estimated_map = self.landmarks

        x, y, theta = self.current_pose

        # FSM LOGIC BELOW
        # 1. this generates semi random escape angles ranging from 90 deg to 270 deg when near a wall
        near_wall = False
        escape_angle = None
        
        limit = 2.2 # trigger turn at 0.3m from wall
        
        if x > limit and np.cos(theta) > 0:
            near_wall = True
            escape_angle = self.wrap_angle(np.pi + np.random.uniform(-0.8, 0.8))
        elif x < -limit and np.cos(theta) < 0:
            near_wall = True
            escape_angle = self.wrap_angle(0.0 + np.random.uniform(-0.8, 0.8))
        elif y > limit and np.sin(theta) > 0:
            near_wall = True
            escape_angle = self.wrap_angle(-np.pi/2 + np.random.uniform(-0.8, 0.8))
        elif y < -limit and np.sin(theta) < 0:
            near_wall = True
            escape_angle = self.wrap_angle(np.pi/2 + np.random.uniform(-0.8, 0.8))

        # 2. landmark avoidance
        near_landmark = False
        for landmark_id, (r_meas, phi_meas) in sensors["observed_landmarks"].items():
            # turn when r measured is less than 0.5 and heading is in 90 deg cone
            if r_meas < 0.5 and abs(phi_meas) < 0.8:
                near_landmark = True
                if escape_angle is None:
                    # turn away from box
                    if phi_meas > 0:
                        # box is to our left so we turn right
                        escape_angle = self.wrap_angle(theta - 1.57)
                    else:
                        # box is to our right so we turn left
                        escape_angle = self.wrap_angle(theta + 1.57)
                break

        # 3. actual state machine
        if self.robot_state == "EXPLORE":
            # drive straight
            control_dict["left_motor"] = 4.0
            control_dict["right_motor"] = 4.0

            # when near landmark set state to avoid and wrap turn target angle
            if near_wall or near_landmark:
                self.robot_state = "AVOID"
                self.turn_target = escape_angle

        elif self.robot_state == "AVOID":
            # wrap target angle
            angle_diff = self.wrap_angle(self.turn_target - theta)
            
            if angle_diff > 0:
                # target is to the left so spin ccw
                control_dict["left_motor"] = -3.0
                control_dict["right_motor"] = 3.0
            else:
                # target is to the Right so spin cw
                control_dict["left_motor"] = 3.0
                control_dict["right_motor"] = -3.0

            # exit AVOID state when facing the target angle
            if abs(angle_diff) < 0.15:
                self.robot_state = "EXPLORE"

        # update pyplot every 10 frames
        self.step_count += 1
        if self.step_count % 10 == 0:
            self.update_plot()

        print(f"POSE: {estimated_pose}")
        print(f"MAP: {estimated_map}")

        return control_dict, estimated_pose, estimated_map