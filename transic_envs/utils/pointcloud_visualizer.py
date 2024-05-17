try:
    import open3d as o3d
except ImportError:
    print("[WARN] open3d not installed, PointCloudVisualizer will not work. Skipping")


class PointCloudVisualizer:
    def __init__(self) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.o3d_pc = o3d.geometry.PointCloud()
        self._initialized = False

    def __call__(self, cloud):
        self.o3d_pc.points = o3d.utility.Vector3dVector(cloud)
        if not self._initialized:
            self.vis.add_geometry(self.o3d_pc)
            self._initialized = True
        self.vis.update_geometry(self.o3d_pc)
        self.vis.update_renderer()
        self.vis.poll_events()
