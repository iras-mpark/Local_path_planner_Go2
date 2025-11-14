# Local_path_planner_Go2

The project is influenced by https://github.com/jizhang-cmu/autonomy_stack_go2

## Local sliding map launcher

To run the LiDAR planner with the short-horizon sliding map (odometry anchored) pipeline:

```bash
ros2 launch local_lidar_planner local_lidar_planner_sliding.launch.xml
```

Tune the `sliding_*` arguments in the launch file to match your LiDAR topic, frame tree, and desired accumulation window.
