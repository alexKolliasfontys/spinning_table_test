#!/usr/bin/env python3

import subprocess
import re
import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

@dataclass
class Pose:
    """Data class to store pose information"""
    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

class KalmanFilter:
    """Simple 1D Kalman Filter for position prediction"""
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-4):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State: [position, velocity]
        self.state = np.array([0.0, 0.0])
        self.covariance = np.eye(2) * 1000  # High initial uncertainty
        
        # State transition matrix (assuming constant velocity)
        self.F = np.array([[1.0, 1.0],  # dt will be multiplied later
                          [0.0, 1.0]])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise
        self.Q = np.array([[0.25, 0.5],
                          [0.5, 1.0]])
        
        # Measurement noise
        self.R = np.array([[self.measurement_variance]])
    
    def predict(self, dt: float):
        """Predict next state"""
        self.F[0, 1] = dt  # Update state transition with actual dt
        self.Q *= self.process_variance * dt  # Scale process noise by dt
        
        # Predict
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return self.state[0]  # Return predicted position
    
    def update(self, measurement: float):
        """Update with new measurement"""
        # Kalman gain
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update
        y = measurement - self.H @ self.state  # Innovation
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ self.H) @ self.covariance
    
    def predict_future(self, dt: float, steps: int):
        """Predict multiple steps into the future"""
        future_state = self.state.copy()
        F_temp = self.F.copy()
        F_temp[0, 1] = dt
        
        predictions = []
        for _ in range(steps):
            future_state = F_temp @ future_state
            predictions.append(future_state[0])
        
        return predictions

class RegressionPredictor:
    """Polynomial regression predictor"""
    
    def __init__(self, degree=2, window_size=10):
        self.degree = degree
        self.window_size = window_size
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=degree)
        self.timestamps = deque(maxlen=window_size)
        self.values = deque(maxlen=window_size)
    
    def update(self, timestamp: float, value: float):
        """Add new data point"""
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def predict_future(self, future_timestamps):
        """Predict values at future timestamps"""
        if len(self.timestamps) < 3:
            return [self.values[-1]] * len(future_timestamps) if self.values else [0] * len(future_timestamps)
        
        # Prepare training data
        X = np.array(self.timestamps).reshape(-1, 1)
        y = np.array(self.values)
        
        # Normalize timestamps to start from 0
        t_start = X[0, 0]
        X_norm = X - t_start
        
        # Transform features
        X_poly = self.poly_features.fit_transform(X_norm)
        
        # Fit model
        self.model.fit(X_poly, y)
        
        # Predict future
        future_X = np.array(future_timestamps).reshape(-1, 1) - t_start
        future_X_poly = self.poly_features.transform(future_X)
        predictions = self.model.predict(future_X_poly)
        
        return predictions

class PoseReceiver:
    """Class to receive and process pose data from Ignition Gazebo"""
    
    def __init__(self, model_name="20x20_Profile", hz=30):
        self.model_name = model_name
        self.hz = hz
        self.dt = 1.0 / hz
        
        # Data storage
        self.poses = deque(maxlen=1000)  # Store last 1000 poses
        self.running = False
        
        # Kalman filters for each coordinate
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        self.kf_z = KalmanFilter()
        
        # Regression predictors
        self.reg_x = RegressionPredictor()
        self.reg_y = RegressionPredictor()
        self.reg_z = RegressionPredictor()
        
        # Thread for data collection
        self.thread = None
    
    def parse_pose_output(self, output: str) -> Optional[Pose]:
        """Parse the IGN pose output"""
        try:
            # Extract XYZ coordinates
            xyz_match = re.search(r'\[([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\]', output)
            if not xyz_match:
                return None
            
            x, y, z = map(float, xyz_match.groups())
            
            # Extract RPY coordinates
            lines = output.split('\n')
            rpy_line = None
            for i, line in enumerate(lines):
                if 'XYZ' in line and i + 1 < len(lines):
                    rpy_line = lines[i + 1]
                    break
            
            if not rpy_line:
                return None
            
            rpy_match = re.search(r'\[([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\]', rpy_line)
            if not rpy_match:
                return None
            
            roll, pitch, yaw = map(float, rpy_match.groups())
            
            return Pose(
                timestamp=time.time(),
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw=yaw
            )
        except Exception as e:
            print(f"Error parsing pose: {e}")
            return None
    
    def get_pose(self) -> Optional[Pose]:
        """Get current pose using ign command"""
        try:
            cmd = ["ign", "model", "-m", self.model_name, "-p"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            
            if result.returncode == 0:
                return self.parse_pose_output(result.stdout)
            else:
                print(f"IGN command failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("IGN command timeout")
            return None
        except Exception as e:
            print(f"Error executing IGN command: {e}")
            return None
    
    def data_collection_loop(self):
        """Main data collection loop"""
        print(f"Starting pose collection at {self.hz} Hz...")
        
        while self.running:
            start_time = time.time()
            
            # Get current pose
            pose = self.get_pose()
            if pose:
                self.poses.append(pose)
                
                # Update Kalman filters
                if len(self.poses) > 1:
                    dt = pose.timestamp - self.poses[-2].timestamp
                    
                    # Predict and update
                    self.kf_x.predict(dt)
                    self.kf_x.update(pose.x)
                    
                    self.kf_y.predict(dt)
                    self.kf_y.update(pose.y)
                    
                    self.kf_z.predict(dt)
                    self.kf_z.update(pose.z)
                
                # Update regression predictors
                self.reg_x.update(pose.timestamp, pose.x)
                self.reg_y.update(pose.timestamp, pose.y)
                self.reg_z.update(pose.timestamp, pose.z)
                
                print(f"Pose: X={pose.x:.6f}, Y={pose.y:.6f}, Z={pose.z:.6f}, "
                      f"Roll={pose.roll:.6f}, Pitch={pose.pitch:.6f}, Yaw={pose.yaw:.6f}")
            
            # Maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)
    
    def start(self):
        """Start data collection"""
        self.running = True
        self.thread = threading.Thread(target=self.data_collection_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop data collection"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def predict_kalman(self, prediction_time: float) -> Tuple[float, float, float]:
        """Predict position using Kalman filter"""
        if len(self.poses) < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate average dt
        timestamps = [p.timestamp for p in list(self.poses)[-10:]]
        if len(timestamps) > 1:
            avg_dt = np.mean(np.diff(timestamps))
        else:
            avg_dt = self.dt
        
        steps = int(prediction_time / avg_dt)
        
        x_pred = self.kf_x.predict_future(avg_dt, steps)
        y_pred = self.kf_y.predict_future(avg_dt, steps)
        z_pred = self.kf_z.predict_future(avg_dt, steps)
        
        return x_pred[-1] if x_pred else 0.0, y_pred[-1] if y_pred else 0.0, z_pred[-1] if z_pred else 0.0
    
    def predict_regression(self, prediction_time: float) -> Tuple[float, float, float]:
        """Predict position using regression"""
        if len(self.poses) < 3:
            return 0.0, 0.0, 0.0
        
        current_time = time.time()
        future_time = current_time + prediction_time
        
        x_pred = self.reg_x.predict_future([future_time])[0]
        y_pred = self.reg_y.predict_future([future_time])[0]
        z_pred = self.reg_z.predict_future([future_time])[0]
        
        return x_pred, y_pred, z_pred
    
    def get_current_pose(self) -> Optional[Pose]:
        """Get the most recent pose"""
        return self.poses[-1] if self.poses else None
    
    def plot_trajectory(self):
        """Plot the trajectory and predictions"""
        if len(self.poses) < 10:
            print("Not enough data for plotting")
            return
        
        poses_list = list(self.poses)
        timestamps = [p.timestamp for p in poses_list]
        x_vals = [p.x for p in poses_list]
        y_vals = [p.y for p in poses_list]
        z_vals = [p.z for p in poses_list]
        
        # Normalize timestamps
        t_start = timestamps[0]
        t_norm = [(t - t_start) for t in timestamps]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # X vs time
        axes[0, 0].plot(t_norm, x_vals, 'b-', label='Actual X')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('X Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Y vs time
        axes[0, 1].plot(t_norm, y_vals, 'g-', label='Actual Y')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Y Position (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Z vs time
        axes[1, 0].plot(t_norm, z_vals, 'r-', label='Actual Z')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Z Position (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 3D trajectory
        axes[1, 1].plot(x_vals, y_vals, 'b-', label='XY Trajectory')
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate usage"""
    # Create pose receiver
    receiver = PoseReceiver(model_name="20x20_Profile", hz=20)  # 20 Hz
    
    try:
        # Start data collection
        receiver.start()
        
        # Let it collect data for a while
        print("Collecting pose data... Press Ctrl+C to stop")
        
        while True:
            time.sleep(5)  # Print predictions every 5 seconds
            
            if len(receiver.poses) > 10:
                # Get current pose
                current_pose = receiver.get_current_pose()
                if current_pose:
                    print(f"\nCurrent pose: X={current_pose.x:.6f}, Y={current_pose.y:.6f}, Z={current_pose.z:.6f}")
                
                # Predict 1 second into the future
                prediction_time = 1.0
                
                # Kalman filter prediction
                kf_x, kf_y, kf_z = receiver.predict_kalman(prediction_time)
                print(f"Kalman prediction (+{prediction_time}s): X={kf_x:.6f}, Y={kf_y:.6f}, Z={kf_z:.6f}")
                
                # Regression prediction
                reg_x, reg_y, reg_z = receiver.predict_regression(prediction_time)
                print(f"Regression prediction (+{prediction_time}s): X={reg_x:.6f}, Y={reg_y:.6f}, Z={reg_z:.6f}")
    
    except KeyboardInterrupt:
        print("\nStopping data collection...")
        receiver.stop()
        
        # Plot results
        receiver.plot_trajectory()

if __name__ == "__main__":
    main()