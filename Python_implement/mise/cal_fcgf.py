import os
import glob
import open3d as o3d
import numpy as np
import torch
import MinkowskiEngine as ME
from mise.pointcloud import make_point_cloud
from mise.fcgf import ResUNetBN2C as FCGF


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    """
    Extracts FCGF features.
    Args:
        model (FCGF model instance): model used to inferr the features
        xyz (torch tensor): coordinates of the point clouds [N,3]
        rgb (torch tensor): colors, must be in range (0,1) [N,3]
        normal (torch tensor): normal vectors, must be in range (-1,1) [N,3]
        voxel_size (float): voxel size for the generation of the saprase tensor
        device (torch device): which device to use, cuda or cpu
        skip_check (bool): if true skip rigorous check (to speed up)
        is_eval (bool): flag for evaluation mode
    Returns:
        return_coords (torch tensor): return coordinates of the points after the voxelization [m,3] (m<=n)
        features (torch tensor): per point FCGF features [m,c]
    """

    if is_eval:
        model.eval()

    if not skip_check:
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        if rgb is not None:
            assert N == len(rgb)
            assert rgb.shape[1] == 3
            if np.any(rgb > 1):
                raise ValueError('Invalid color. Color must range from [0, 1]')

        if normal is not None:
            assert N == len(normal)
            assert normal.shape[1] == 3
            if np.any(normal > 1):
                raise ValueError('Invalid normal. Normal must range from [-1, 1]')

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)
    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    return return_coords, model(stensor).F


def process_3dmatch(voxel_size=0.05):
    root = "../data/3DMatch/threedmatch/"
    save_path = "../data/3DMatch/threedmatch_feat/"
    pcd_list = os.listdir(root)
    for pcd_path in pcd_list:
        if pcd_path.endswith('.npz') is False:
            continue
        full_path = os.path.join(root, pcd_path)
        data = np.load(full_path)
        pcd = data['pcd']
        if pcd.shape[0] == 0:
            print(f"{full_path} error: do not have any points.")
            continue

        xyz_down, features = extract_features(
            model,
            xyz=pcd,
            rgb=None,
            normal=None,
            voxel_size=voxel_size,
            skip_check=True,
        )
        np.savez_compressed(
            os.path.join(save_path, pcd_path.replace('.npz', '_fcgf.npz')),
            points=pcd.astype(np.float32),
            xyz=xyz_down.astype(np.float32),
            feature=features.detach().cpu().numpy().astype(np.float32)
        )


def process_3dmatch_test(voxel_size=0.05):
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    for scene in scene_list:
        scene_path = os.path.join("../data/3DMatch/fragments/", scene)
        pcd_list = os.listdir(scene_path)
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            pcd = o3d.io.read_point_cloud(full_path)

            xyz_down, features = extract_features(
                model,
                xyz=np.array(pcd.points),
                rgb=None,
                normal=None,
                voxel_size=voxel_size,
                skip_check=True,
            )

            np.savez_compressed(
                full_path.replace('.ply', '_fcgf'),
                points=np.array(pcd.points).astype(np.float32),
                xyz=xyz_down.astype(np.float32),
                feature=features.detach().cpu().numpy().astype(np.float32)
            )
            print(full_path)


def process_redwood(voxel_size=0.05):
    scene_list = [
        'livingroom1-simulated',
        'livingroom2-simulated',
        'office1-simulated',
        'office2-simulated'
    ]
    for scene in scene_list:
        scene_path = os.path.join("../data/Augmented_ICL-NUIM/", scene + '/fragments')
        pcd_list = os.listdir(scene_path)
        pcd_list = [x for x in pcd_list if x.endswith('.ply')]
        # pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            pcd = o3d.io.read_point_cloud(full_path)

            # voxel downsample and compute fcgf descriptor
            xyz_down, features = extract_features(
                model,
                xyz=np.array(pcd.points),
                rgb=None,
                normal=None,
                voxel_size=voxel_size,
                skip_check=True,
            )
            
            # save the data for training.
            np.savez_compressed(
                full_path.replace('.ply', '_fcgf'),
                points=np.array(pcd.points).astype(np.float32),
                xyz=xyz_down.astype(np.float32),
                feature=features.detach().cpu().numpy().astype(np.float32),
            )
            print(full_path)


kitti_cache = {}
kitti_icp_cache = {}


def process_kitti(voxel_size=0.30, split='train'):
    def odometry_to_positions(odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_video_odometry(root, drive, indices=None, ext='.txt', return_all=False):
        data_path = root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def _get_velodyne_fn(root, drive, t):
        fname = root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    MIN_DIST = 10
    root = '/media/zxy/4203AF93BB0444E6/KITTI_dataset/dataset'
    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T

    subset_names = open(f'split/{split}_kitti.txt').read().split()
    files = []
    for dirname in subset_names:
        drive_id = int(dirname)
        # inames = get_all_scan_ids(root, drive_id)
        # for start_time in inames:
        #     for time_diff in range(2, max_time_diff):
        #         pair_time = time_diff + start_time
        #         if pair_time in inames:
        #             files.append((drive_id, start_time, pair_time))
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = get_video_odometry(root, drive_id, return_all=True)
        all_pos = np.array([odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
        pdist = np.sqrt(pdist.sum(-1))
        more_than_10 = pdist > MIN_DIST
        curr_time = inames[0]
        while curr_time in inames:
            next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0:
                curr_time += 1
            else:
                next_time = next_time[0] + curr_time - 1

            if next_time in inames:
                files.append((drive_id, curr_time, next_time))
                curr_time = next_time + 1
        # Remove problematic sequence
        for item in [
            (8, 15, 58),
        ]:
            if item in files:
                files.pop(files.index(item))

    # begin extracting features
    for idx in range(len(files)):
        drive = files[idx][0]
        t0, t1 = files[idx][1], files[idx][2]
        all_odometry = get_video_odometry(root, drive, [t0, t1])
        positions = [odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = _get_velodyne_fn(root, drive, t0)
        fname1 = _get_velodyne_fn(root, drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)
        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = root + '/icp/' + key + '.npy'
        save_pointcloud_path = root + '/point_cloud/drive' + str(drive)
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                # work on the downsampled xyzs, 0.05m == 5cm
                corrds, sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                corrds, sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T) @ np.linalg.inv(velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0_ori = make_point_cloud(xyz0[sel0])
                pcd0 = make_point_cloud(xyz0_t)
                pcd1 = make_point_cloud(xyz1[sel1])
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
                M2 = M @ reg.transformation
                #o3d.visualization.draw_geometries([pcd0, pcd1])
                if not os.path.exists(save_pointcloud_path):
                    os.makedirs(save_pointcloud_path)
                o3d.io.write_point_cloud(save_pointcloud_path + '/'+str(t0) + '.ply', pcd0_ori)
                o3d.io.write_point_cloud(save_pointcloud_path + '/'+str(t1) + '.ply', pcd1)
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]

        xyz_down0, features0 = extract_features(
            model,
            xyz=xyz0,
            rgb=None,
            normal=None,
            voxel_size=voxel_size,
            skip_check=True,
        )
        xyz_down1, features1 = extract_features(
            model,
            xyz=xyz1,
            rgb=None,
            normal=None,
            voxel_size=voxel_size,
            skip_check=True,
        )
        filename = f"{root}/fcgf_{split}/drive{drive}-pair{t0}_{t1}"
        np.savez_compressed(
            filename,
            xyz0=xyz_down0.astype(np.float32),
            xyz1=xyz_down1.astype(np.float32),
            features0=features0.detach().cpu().numpy().astype(np.float32),
            features1=features1.detach().cpu().numpy().astype(np.float32),
            gt_trans=M2
        )
        print(filename)


if __name__ == '__main__':
    # model = FCGF(
    #     1,
    #     32,
    #     bn_momentum=0.05,
    #     conv1_kernel_size=7,
    #     normalize_feature=True
    # ).cuda()
    # # 3DMatch: http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-3dmatch-v0.05.pth
    # # KITTI: http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-kitti-v0.3.pth
    # checkpoint = torch.load("ResUNetBN2C-feat32-3dmatch-v0.05.pth")
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # process_3dmatch(voxel_size=0.05)
    # process_3dmatch_test(voxel_size=0.05)
    # process_redwood(voxel_size=0.05)

    model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=5,
        normalize_feature=True
    ).cuda()
    checkpoint = torch.load("ResUNetBN2C-feat32-kitti-v0.3.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    process_kitti(voxel_size=0.30, split='test')
