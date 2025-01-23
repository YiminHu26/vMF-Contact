import numpy as np
import scipy.linalg

class GraspKalmanFilter:
    """
    Kalman filter for grasp pose tracking in 3D space.
    
    The 14-dimensional state space:

        x, y, z, ax, ay, az, w, vx, vy, vz, vax, vay, vaz, vw

    Tracks the grasp baseline position (x, y, z), approach direction (ax, ay, az),
    grasp width (w), and their respective velocities.
    """

    def __init__(self):
        ndim, dt = 7, 1.0  # 7 state components (pos, approach, width)
        
        # Motion model: identity matrix with velocity update
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """
        Create track from an unassociated measurement.
        measurement: [x, y, z, ax, ay, az, w]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[6],  # w affects uncertainty
            2 * self._std_weight_position * measurement[6],
            2 * self._std_weight_position * measurement[6],
            1e-2, 1e-2, 1e-2, 
            2 * self._std_weight_position * measurement[6],
            10 * self._std_weight_velocity * measurement[6],
            10 * self._std_weight_velocity * measurement[6],
            10 * self._std_weight_velocity * measurement[6],
            1e-5, 1e-5, 1e-5,
            10 * self._std_weight_velocity * measurement[6]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state."""
        std_pos = [
            self._std_weight_position * mean[6],
            self._std_weight_position * mean[6],
            self._std_weight_position * mean[6],
            1e-2, 1e-2, 1e-2, 
            self._std_weight_position * mean[6]
        ]
        std_vel = [
            self._std_weight_velocity * mean[6],
            self._std_weight_velocity * mean[6],
            self._std_weight_velocity * mean[6],
            1e-5, 1e-5, 1e-5,
            self._std_weight_velocity * mean[6]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[6],
            self._std_weight_position * mean[6],
            self._std_weight_position * mean[6],
            1e-1, 1e-1, 1e-1,
            self._std_weight_position * mean[6]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Update state with a new measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:3], covariance[:3, :3]
            measurements = measurements[:, :3]

        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
