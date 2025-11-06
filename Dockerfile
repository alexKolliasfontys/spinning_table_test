# ===== Dockerfile =====
FROM osrf/ros:humble-desktop-full

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-gazebo-ros \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /root/ws_moveit2

# Copy source files
COPY src ./src

# Build the workspace
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Source the environment automatically
RUN echo "source /opt/ros/humble/setup.sh" >> ~/.bashrc && \
    echo "source /root/ws_moveit2/install/setup.sh" >> ~/.bashrc

# Default command: open a bash shell
CMD ["bash"]
