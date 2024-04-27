#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import open3d as o3d
import numpy as np
import PIL.Image as PILImage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_image', type=str, default=None)
parser.add_argument('--in_depth', type=str, default=None)
parser.add_argument('--mode', type=str, choices=['pcd', 'mesh'], default='mesh')
parser.add_argument('--out_pcd', type=str, default=None)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--shift', type=float, default=0.0)
parser.add_argument('--near', type=float, default=1e-5)
parser.add_argument('--far', type=float, default=1e5)
parser.add_argument('--span', type=float, default=360.0)
args = parser.parse_args()

in_image = args.in_image
in_depth = args.in_depth
out_pcd = args.out_pcd

image = PILImage.open(in_image).convert('RGB')
image = np.array(image)

if in_depth.startswith('dummy://'):
    depth_value = float(in_depth[len('dummy://'):])
    depth = np.full((80, 320), depth_value)
else:
    depth = PILImage.open(in_depth).convert('L')
    depth = np.array(depth).astype(np.float32).clip(1,255)
    depth = depth / 255
    depth = depth * args.scale + args.shift
    depth = 1 / depth
    depth = depth.clip(args.near, args.far)

H, W = depth.shape
LONG = args.span / 360 * np.pi * 2
LAT = LONG / W * H

image_index = np.transpose(np.mgrid[:H, :W], (1,2,0))
image_coord = image_index[:,:,::-1] + 0.5
long = image_coord[:,:,0] / W * LONG
lat = image_coord[:,:,1] / H * LAT - LAT / 2
h = np.tan(-lat)

y = h * depth
x = depth * np.cos(long)
z = depth * np.sin(long)
world_coord = np.stack([x,y,z], axis=-1)

if args.mode == 'pcd':
    image = image.astype(np.float32)
    image = image / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coord.reshape(-1,3))
    pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1,3))
    o3d.io.write_point_cloud(out_pcd, pcd)
elif args.mode == 'mesh':
    v00 = image_index
    v01 = image_index + np.array([0,1])
    v10 = image_index + np.array([1,0])
    v11 = image_index + np.array([1,1])
    for arr in v01, v10, v11:  # circular connection
        arr[..., 0] %= H
        arr[..., 1] %= W
    v00, v01, v10, v11 = [  # do not circular vertically [H, W, 2]
        arr[:-1, :-1] for arr in [v00, v01, v10, v11]
    ]
    tri_0 = np.stack([v00, v10, v11], axis=-2)  # [H, W, 3, 2]
    tri_1 = np.stack([v00, v11, v01], axis=-2)  # [H, W, 3, 2]
    tri = np.stack([tri_0, tri_1], axis=-3)  # [H, W, 2 (#tri in a square), 3 (#vert in a tri), 2 (ij)]
    tri_f = tri[..., 0] * W + tri[..., 1]  # [H, W, 2, 3]

    uv = tri[..., ::-1] + 0.5  # [H, W, 2, 3, 2]
    uv[..., 0] /= W
    uv[..., 1] /= H

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(world_coord.reshape(-1,3))
    mesh.triangles = o3d.utility.Vector3iVector(tri_f.reshape(-1,3))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv.reshape(-1,2))
    mesh.textures = [o3d.geometry.Image(image)]
    o3d.io.write_triangle_mesh(out_pcd, mesh)
