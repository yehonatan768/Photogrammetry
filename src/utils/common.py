from pathlib import Path
import open3d as o3d


def header(title: str) -> None:
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def o3d_module():
    import open3d as o3d
    return o3d


def write_ply(path: Path, pcd: "o3d.geometry.PointCloud") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    return path
