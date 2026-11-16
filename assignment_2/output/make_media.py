'''Build the figures, stills, and videos used by the assignment report.

Run from assignment_2. Panda3D is opened offscreen, so this does not need a visible window.
'''
import json
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'audio-library-name null')
loadPrcFileData('', 'win-size 1280 720')
loadPrcFileData('', 'sync-video false')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from file_io import BVHMotion
from task1_motion_editing import interpolation, concatenate_two_motions
from panda3d.core import ClockObject, Filename, LVector3, PNMImage, getModelPath
from Viewer.viewer import SimpleViewer
from Viewer.controller import Controller
from task2_motion_matching import (
    CharacterController, Database, InteractiveUpdate, feature_mapping,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
FIG = os.path.join(OUT, 'figures')
VID = os.path.join(OUT, 'videos')
os.makedirs(FIG, exist_ok=True)
os.makedirs(VID, exist_ok=True)

DT = 1.0 / 60.0
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
})


def geodesic(q_a, q_b):
    dots = np.abs(np.sum(q_a * q_b, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dots)


def build_keyframes(motion, time_step, target_step):
    length = motion.local_joint_positions.shape[0]
    keyframes = np.arange(0, length, time_step)
    positions, rotations = [], []
    previous = 0
    for current in keyframes[1:]:
        positions.append(interpolation(
            motion.local_joint_positions[previous],
            motion.local_joint_positions[current],
            target_step - 1, 'linear'))
        rotations.append(interpolation(
            motion.local_joint_rotations[previous],
            motion.local_joint_rotations[current],
            target_step - 1, 'slerp'))
        previous = current
    pos = np.concatenate(positions, axis=0)
    rot = np.concatenate(rotations, axis=0)
    return pos, rot


def root_yaw(quat):
    forward = R.from_quat(quat).apply(np.array([0.0, 0.0, 1.0]))
    return float(np.arctan2(forward[0], forward[2]))


def match_indices(walk, run, last_frame_index=40, start_frame_indx=0, searching_frames=20):
    s1 = max(0, last_frame_index - searching_frames)
    e1 = min(len(walk.local_joint_rotations), last_frame_index + searching_frames)
    s2 = max(0, start_frame_indx - searching_frames)
    e2 = min(len(run.local_joint_rotations), start_frame_indx + searching_frames)
    a = walk.local_joint_rotations[s1:e1].copy()
    b = run.local_joint_rotations[s2:e2].copy()
    a[a[..., 3] < 0] *= -1
    b[b[..., 3] < 0] *= -1
    src = a.reshape(len(a), -1)
    dst = b.reshape(len(b), -1)
    sim = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmin(sim)), sim.shape)
    return sim, s1 + int(i), s2 + int(j)


def fresh_run(walk):
    run = BVHMotion('data/motion_running.bvh')
    run.adjust_joint_name(walk.joint_name)
    return run


def joint_step(rot):
    steps = []
    for t in range(1, len(rot)):
        delta = R.from_quat(rot[t]) * R.from_quat(rot[t - 1]).inv()
        steps.append(delta.magnitude().max())
    return np.array(steps)


def main_numbers():
    walk = BVHMotion('data/motion_walking.bvh')
    run = fresh_run(walk)
    stats = {}

    # Part 1 speed table and sparse-key error.
    speed_rows = []
    for target in (5, 10, 20, 30):
        pos, rot = build_keyframes(walk, 10, target)
        covered = (np.arange(0, len(walk.local_joint_positions), 10)[-1]) / 60.0
        play = len(pos) / 60.0
        speed_rows.append({
            'time_step': 10,
            'target_step': target,
            'frames': int(len(pos)),
            'playback_s': round(play, 2),
            'source_s': round(covered, 2),
            'speed': round(covered / play, 2),
        })
    stats['speed_rows'] = speed_rows

    errors = []
    for step in (5, 10, 20, 40):
        keys = np.arange(0, len(walk.local_joint_rotations), step)
        errs = []
        for k0, k1 in zip(keys[:-1], keys[1:]):
            mid = (k0 + k1) // 2
            guessed = interpolation(
                walk.local_joint_rotations[k0],
                walk.local_joint_rotations[k1], 2, 'slerp')[1]
            errs.append(geodesic(guessed, walk.local_joint_rotations[mid]).mean())
        errors.append({'time_step': int(step), 'midpoint_error_deg': round(float(np.degrees(np.mean(errs))), 2)})
    stats['key_errors'] = errors

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for target, color in zip((5, 10, 20, 30), ('#d95f02', '#1b9e77', '#7570b3', '#e7298a')):
        pos, _ = build_keyframes(walk, 10, target)
        time = np.arange(len(pos)) / 60.0
        ax.plot(time, pos[:, 0, 2], color=color, lw=1.8, label=f'target_step = {target}')
    ax.set_xlabel('Playback time (s)')
    ax.set_ylabel('Root position Z (m)')
    ax.set_title('Same keys, different in-between counts')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'part1_speed.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot([r['time_step'] for r in errors], [r['midpoint_error_deg'] for r in errors],
            marker='o', color='#1b9e77', lw=1.8)
    ax.set_xlabel('Key interval (frames)')
    ax.set_ylabel('Midpoint error (degrees)')
    ax.set_title('Slerp midpoint versus the original pose')
    fig.savefig(os.path.join(FIG, 'part1_key_error.png'))
    plt.close(fig)

    # Part 2.
    sim, real_i, real_j = match_indices(walk, run)
    stats['real_i'] = int(real_i)
    stats['real_j'] = int(real_j)
    stats['sim_min'] = round(float(sim.min()), 3)
    stats['yaw_walk'] = round(float(np.degrees(root_yaw(walk.local_joint_rotations[real_i, 0]))), 2)
    stats['yaw_run'] = round(float(np.degrees(root_yaw(run.local_joint_rotations[real_j, 0]))), 2)
    raw_jump = float(np.linalg.norm(
        walk.local_joint_positions[40, 0] - run.local_joint_positions[0, 0]))
    stats['raw_jump_m'] = round(raw_jump, 2)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(sim, origin='lower', aspect='auto', cmap='viridis_r')
    ax.scatter([real_j - (0)], [real_i - (40 - 20)], c='white', s=40, marker='x')
    # window coordinates of the minimum
    wi, wj = np.unravel_index(int(np.argmin(sim)), sim.shape)
    ax.scatter([wj], [wi], c='white', s=36, marker='x')
    ax.set_xlabel('Running window frame')
    ax.set_ylabel('Walking window frame')
    ax.set_title('Pose distance between the two search windows')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Quaternion L2')
    fig.savefig(os.path.join(FIG, 'part2_sim_matrix.png'))
    plt.close(fig)

    clips = {}
    for method, between in (('interpolation', 8), ('inertialization', 8)):
        motion = concatenate_two_motions(walk, fresh_run(walk), 40, 0, between, method=method)
        clips[method] = motion
    # raw splice at the requested indices, for the failure case
    raw = walk.raw_copy()
    raw_run = fresh_run(walk)
    raw.local_joint_positions = np.concatenate([
        walk.local_joint_positions[:40], raw_run.local_joint_positions[0:]], axis=0)
    raw.local_joint_rotations = np.concatenate([
        walk.local_joint_rotations[:40], raw_run.local_joint_rotations[0:]], axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label, motion, color in (
        ('Direct splice', raw, '#d95f02'),
        ('Interpolation + root Hermite', clips['interpolation'], '#1b9e77'),
        ('Inertialization', clips['inertialization'], '#7570b3'),
    ):
        root = motion.local_joint_positions[:, 0]
        ax.plot(root[:, 0], root[:, 2], color=color, lw=1.8, label=label)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Root X (m)')
    ax.set_ylabel('Root Z (m)')
    ax.set_title('Root path through the transition')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'part2_root_path.png'))
    plt.close(fig)

    # How long should the blend be?
    angle = geodesic(
        walk.local_joint_rotations[real_i],
        run.local_joint_rotations[real_j]).mean()
    source_step = max(joint_step(walk.local_joint_rotations).max(),
                      joint_step(run.local_joint_rotations).max())
    rows = []
    for between in (2, 4, 8, 12, 20, 30, 45):
        motion = concatenate_two_motions(
            walk, fresh_run(walk), 40, 0, between, method='interpolation')
        # The blend occupies [real_i, real_i + between).
        root = motion.local_joint_positions[:, 0]
        travel = float(np.linalg.norm(root[real_i + between - 1] - root[real_i]))
        steps = joint_step(motion.local_joint_rotations[real_i:real_i + between + 1])
        rows.append({
            'between_frames': between,
            'duration_s': round(between * DT, 3),
            'root_travel_m': round(travel, 3),
            'max_joint_step_deg': round(float(np.degrees(steps.max())), 2),
        })
    stats['blend_rows'] = rows
    stats['pose_distance_deg'] = round(float(np.degrees(angle)), 2)
    stats['source_max_step_deg'] = round(float(np.degrees(source_step)), 2)
    # Smallest blend whose fastest joint is no faster than the source clips.
    limit = float(np.degrees(source_step))
    suggested = next(r['between_frames'] for r in rows if r['max_joint_step_deg'] <= limit + 0.05)
    stats['suggested_between'] = int(suggested)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = [r['between_frames'] for r in rows]
    ax.plot(xs, [r['max_joint_step_deg'] for r in rows], marker='o', color='#1b9e77', label='Fastest joint in the blend')
    ax.axhline(limit, color='#d95f02', ls='--', label=f'Source clip maximum ({limit:.1f}°/frame)')
    ax.set_xlabel('In-between frames')
    ax.set_ylabel('Degrees per frame')
    ax.set_title('Blend length versus how hard the pose has to move')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'part2_blend_length.png'))
    plt.close(fig)

    # Keep the two showcase motions for rendering.
    np.save(os.path.join(OUT, 'clip_interp_pos.npy'), clips['interpolation'].local_joint_positions)
    np.save(os.path.join(OUT, 'clip_interp_rot.npy'), clips['interpolation'].local_joint_rotations)
    np.save(os.path.join(OUT, 'clip_inert_pos.npy'), clips['inertialization'].local_joint_positions)
    np.save(os.path.join(OUT, 'clip_inert_rot.npy'), clips['inertialization'].local_joint_rotations)
    stats['joint_name'] = walk.joint_name
    stats['interp_frames'] = int(len(clips['interpolation'].local_joint_positions))
    stats['inert_frames'] = int(len(clips['inertialization'].local_joint_positions))

    # Part 1 motions for stills, saved cheaply by rebuilding later.
    with open(os.path.join(OUT, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats, walk


def feature_study(stats):
    names = list(feature_mapping.keys())
    motions = [BVHMotion('data/motion_walking_long.bvh'), BVHMotion('data/push.bvh')]
    db = Database(motions, feature_mapping, names, [1.0] * len(names))
    cols = {}
    start = 0
    for name in names:
        dim = feature_mapping[name]
        cols[name] = slice(start, start + dim)
        start += dim
    walk_end = motions[0].motion_data.shape[0]
    cands = np.arange(100, walk_end - 80, 40)

    def evaluate(subset, weights, yaw_deg):
        data = np.concatenate(
            [db.normalized_feature[:, cols[n]] * w for n, w in zip(subset, weights)], axis=1)
        tree = cKDTree(data)
        angs, foots, vels = [], [], []
        yaw = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        for fi in cands:
            pos = db.features[fi, cols['trajectoryPos2D']].reshape(3, 2)
            direc = db.features[fi, cols['trajectoryRot2D']].reshape(3, 2)
            def spin(v):
                x, z = v[:, 0], v[:, 1]
                return np.stack([c * x + s * z, -s * x + c * z], axis=1)
            pos_t = spin(pos).reshape(-1)
            dir_t = spin(direc).reshape(-1)
            override = {
                'trajectoryPos2D': (pos_t - db.features_mean[cols['trajectoryPos2D']]) / db.features_std[cols['trajectoryPos2D']],
                'trajectoryRot2D': (dir_t - db.features_mean[cols['trajectoryRot2D']]) / db.features_std[cols['trajectoryRot2D']],
            }
            parts = []
            for n, w in zip(subset, weights):
                block = override[n] if n in override else db.normalized_feature[fi, cols[n]]
                parts.append(block * w)
            idx = int(tree.query(np.concatenate(parts), k=1)[1])
            desired = pos_t.reshape(3, 2)[-1]
            got = db.features[idx, cols['trajectoryPos2D']].reshape(3, 2)[-1]
            na, nb = np.linalg.norm(desired), np.linalg.norm(got)
            if na < 1e-6 or nb < 1e-6:
                angs.append(90.0)
            else:
                angs.append(np.degrees(np.arccos(np.clip(np.dot(desired, got) / (na * nb), -1, 1))))
            for store, key in ((foots, 'Pos'), (vels, 'Vel')):
                a = np.concatenate([db.features[fi, cols[f'lFoot{key}']], db.features[fi, cols[f'rFoot{key}']]])
                b = np.concatenate([db.features[idx, cols[f'lFoot{key}']], db.features[idx, cols[f'rFoot{key}']]])
                store.append(np.linalg.norm(a - b))
        return {
            'turn_deg': round(float(np.median(angs)), 1),
            'foot_m': round(float(np.median(foots)), 3),
            'foot_vel': round(float(np.median(vels)), 3),
        }

    presets = [
        ('Trajectory only', ['trajectoryPos2D', 'trajectoryRot2D'], [1, 1]),
        ('Feet only', ['lFootPos', 'rFootPos'], [0.75, 0.75]),
        ('Feet + hands', ['lFootPos', 'rFootPos', 'lHandPos', 'rHandPos'], [0.75, 0.75, 0.75, 0.75]),
        ('Feet position + velocity', ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel'], [0.75, 0.75, 1, 1]),
        ('Feet + path (chosen)', ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'trajectoryPos2D'], [0.75, 0.75, 1, 1, 1]),
        ('Feet + path + facing', ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'trajectoryPos2D', 'trajectoryRot2D'], [0.75, 0.75, 1, 1, 1, 1]),
        ('Above + hip velocity', ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'hipVel', 'trajectoryPos2D', 'trajectoryRot2D'], [0.75, 0.75, 1, 1, 0.5, 1, 1]),
    ]
    rows = []
    for label, subset, weights in presets:
        measured = evaluate(subset, weights, 60)
        measured['name'] = label
        measured['n'] = len(subset)
        rows.append(measured)
        print(label, measured)
    stats['features'] = rows

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = [r['name'] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r['turn_deg'] for r in rows], width=0.36, color='#1b9e77', label='Turn error (deg, median)')
    ax.bar(x + 0.18, [r['foot_m'] * 80 for r in rows], width=0.36, color='#d95f02', label='Foot jump × 80 (m)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.set_ylabel('Lower is better')
    ax.set_title('A 60° turn request, searched over the locomotion database')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'part3_features.png'))
    plt.close(fig)

    # Future horizon, with the chosen five features. Horizons stay length 3
    # so the feature width matches the controller, which samples three futures.
    root_pos = db.motion_data[:, 0:3]
    root_rot = R.from_euler('XYZ', db.motion_data[:, 3:6], degrees=True).as_quat()
    foot_slice = slice(0, 12)  # four foot channels, 3 dims each, in the full vector? 
    # The full database was built with every feature, so foot columns are not a prefix.
    # Use the named slices instead.

    def horizon_feature(frames):
        rot_inv = R.from_quat(root_rot).inv()
        blocks = []
        for t in frames:
            idx = np.clip(np.arange(len(root_pos)) + t, 0, len(root_pos) - 1)
            local = rot_inv.apply(root_pos[idx] - root_pos)
            blocks.append(local[:, [0, 2]])
        return np.concatenate(blocks, axis=1)

    subset = ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'trajectoryPos2D']
    weights = [0.75, 0.75, 1.0, 1.0, 1.0]
    horizon_rows = []
    for frames in ([10, 20, 30], [20, 40, 60], [30, 60, 90], [40, 80, 120]):
        raw = horizon_feature(frames)
        mean, std = raw.mean(0), raw.std(0)
        std[std < 0.1] = 0.1
        norm_traj = (raw - mean) / std
        columns = [db.normalized_feature[:, cols[n]] for n in subset[:-1]]
        columns.append(norm_traj)
        data = np.concatenate([col * w for col, w in zip(columns, weights)], axis=1)
        tree = cKDTree(data)
        angs, foots = [], []
        yaw = np.deg2rad(60)
        c, s = np.cos(yaw), np.sin(yaw)
        usable = cands[cands < walk_end - max(frames) - 5]
        for fi in usable:
            base = raw[fi].reshape(3, 2)
            x, z = base[:, 0], base[:, 1]
            spun = np.stack([c * x + s * z, -s * x + c * z], axis=1).reshape(-1)
            query_parts = [db.normalized_feature[fi, cols[n]] * w for n, w in zip(subset[:-1], weights[:-1])]
            query_parts.append(((spun - mean) / std) * weights[-1])
            idx = int(tree.query(np.concatenate(query_parts), k=1)[1])
            desired = spun.reshape(3, 2)[-1]
            got = raw[idx].reshape(3, 2)[-1]
            na, nb = np.linalg.norm(desired), np.linalg.norm(got)
            angs.append(90.0 if na < 1e-6 or nb < 1e-6 else np.degrees(
                np.arccos(np.clip(np.dot(desired, got) / (na * nb), -1, 1))))
            a = np.concatenate([db.features[fi, cols['lFootPos']], db.features[fi, cols['rFootPos']]])
            b = np.concatenate([db.features[idx, cols['lFootPos']], db.features[idx, cols['rFootPos']]])
            foots.append(np.linalg.norm(a - b))
        horizon_rows.append({
            'frames': frames,
            'turn_deg': round(float(np.median(angs)), 1),
            'foot_m': round(float(np.median(foots)), 3),
        })
        print('horizon', frames, horizon_rows[-1])
    stats['horizons'] = horizon_rows
    with open(os.path.join(OUT, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)


class MouseStub:
    def is_button_down(self, *args, **kwargs):
        return False


def open_viewer():
    getModelPath().prependDirectory(Filename.from_os_specific(ROOT))
    viewer = SimpleViewer()
    viewer.setFrameRateMeter(False)
    viewer.mouseWatcherNode = MouseStub()
    globalClock.setMode(ClockObject.MNonRealTime)
    globalClock.setDt(DT)
    return viewer


def show_pose(viewer, joint_name, trans, ori, frame):
    for i, name in enumerate(joint_name):
        viewer.set_joint_position_orientation(name, trans[frame, i], ori[frame, i])


def follow(viewer, root):
    viewer.cameractrl.center = LVector3(float(root[0]), 0.95, float(root[2]))
    viewer.cameractrl.position = LVector3(float(root[0]) + 2.3, 1.85, float(root[2]) - 3.15)
    viewer.cameractrl.look()


def grab(viewer, path):
    for _ in range(2):
        viewer.graphicsEngine.renderFrame()
    image = PNMImage()
    if not viewer.win.getScreenshot(image):
        raise RuntimeError('screenshot failed')
    image.write(Filename(path))


def pnm_to_pil(viewer):
    viewer.graphicsEngine.renderFrame()
    image = PNMImage()
    viewer.win.getScreenshot(image)
    image.write(Filename(os.path.join(OUT, '_frame.png')))
    return Image.open(os.path.join(OUT, '_frame.png')).convert('RGB')


def font(size):
    for path in (
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/Library/Fonts/Arial.ttf',
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def title_frame(text, subtitle):
    image = Image.new('RGB', (1280, 720), (28, 32, 36))
    draw = ImageDraw.Draw(image)
    draw.text((80, 280), text, fill=(245, 245, 242), font=font(54))
    draw.text((80, 370), subtitle, fill=(190, 198, 190), font=font(28))
    return image


def start_video(path, size, fps):
    log = open(path + '.log', 'w')
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        '-movflags', '+faststart', path,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    proc._log = log
    return proc


def write_frame(proc, frame):
    proc.stdin.write(frame.convert('RGB').tobytes())


def finish_video(proc, path):
    proc.stdin.close()
    code = proc.wait()
    proc._log.close()
    if code != 0:
        raise RuntimeError('ffmpeg failed for ' + path)
    print('video', path, os.path.getsize(path))


def render_all(stats):
    viewer = open_viewer()
    walk = BVHMotion('data/motion_walking.bvh')
    joint_name = walk.joint_name

    # Part 1 stills at one second of playback.
    panels = []
    for target in (5, 10, 20, 30):
        pos, rot = build_keyframes(walk, 10, target)
        motion = walk.raw_copy()
        motion.local_joint_positions = pos
        motion.local_joint_rotations = rot
        trans, ori = motion.batch_forward_kinematics()
        frame = min(60, len(trans) - 1)
        show_pose(viewer, joint_name, trans, ori, frame)
        follow(viewer, trans[frame, 0])
        path = os.path.join(FIG, f'part1_t{target}.png')
        grab(viewer, path)
        panels.append(Image.open(path).convert('RGB'))
        print('still', path)
    # Mid-stride of the real-time setting, used as the walking figure.
    pos, rot = build_keyframes(walk, 10, 10)
    motion = walk.raw_copy()
    motion.local_joint_positions = pos
    motion.local_joint_rotations = rot
    trans, ori = motion.batch_forward_kinematics()
    show_pose(viewer, joint_name, trans, ori, 70)
    follow(viewer, trans[70, 0])
    grab(viewer, os.path.join(FIG, 'part1_walk.png'))

    sheet = Image.new('RGB', (1280 * 2, 720 * 2), (255, 255, 255))
    labels = ['target_step = 5  (2.5× speed)', 'target_step = 10  (about real time)',
              'target_step = 20  (about half speed)', 'target_step = 30  (about one third speed)']
    for i, (panel, label) in enumerate(zip(panels, labels)):
        tile = panel.copy()
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 760, 52), fill=(20, 24, 28))
        draw.text((16, 10), label, fill=(245, 245, 242), font=font(28))
        sheet.paste(tile, ((i % 2) * 1280, (i // 2) * 720))
    sheet.save(os.path.join(FIG, 'part1_speeds_sheet.png'))

    # Part 2 transition stills and the video.
    concat_video = start_video(os.path.join(VID, 'motion_concatenation.mp4'), (1280, 720), 30)
    for key, title, subtitle in (
        ('interp', 'Motion concatenation', 'Walk into run, matched poses with a velocity-aware root'),
        ('inert', 'Inertialization', 'The run plays immediately and the pose offset decays'),
    ):
        pos = np.load(os.path.join(OUT, f'clip_{key}_pos.npy'))
        rot = np.load(os.path.join(OUT, f'clip_{key}_rot.npy'))
        motion = walk.raw_copy()
        motion.local_joint_positions = pos
        motion.local_joint_rotations = rot
        trans, ori = motion.batch_forward_kinematics()
        clip_frames = []
        still_at = [0, len(trans) // 3, 2 * len(trans) // 3, len(trans) - 1]
        for frame in range(len(trans)):
            show_pose(viewer, joint_name, trans, ori, frame)
            follow(viewer, trans[frame, 0])
            image = pnm_to_pil(viewer)
            clip_frames.append(image)
            if frame in still_at:
                image.save(os.path.join(FIG, f'part2_{key}_{frame:03d}.png'))
        for image in [title_frame(title, subtitle)] * 24:
            write_frame(concat_video, image)
        for _ in range(3):
            for image in clip_frames:
                write_frame(concat_video, image)
        print(key, 'frames', len(clip_frames))
    finish_video(concat_video, os.path.join(VID, 'motion_concatenation.mp4'))

    # Part 3, driven the same way the interactive demo is driven.
    # A new viewer keeps the controller's camera rig.
    viewer.destroy()
    viewer = open_viewer()
    controller = Controller(viewer)
    names = ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'trajectoryPos2D']
    weights = [0.75, 0.75, 1.0, 1.0, 1.0]
    character = CharacterController(viewer, controller, names, weights)
    viewer.addTask(InteractiveUpdate(viewer, controller, character).update)

    def command(second):
        if second < 3.0:
            vel = np.array([0.0, 0.0, 1.0])
        elif second < 6.0:
            vel = np.array([1.0, 0.0, 1.0])
        elif second < 9.0:
            vel = np.array([0.0, 0.0, 1.0])
        elif second < 11.5:
            vel = np.array([-1.0, 0.0, 1.0])
        else:
            vel = np.zeros(3)
        norm = np.linalg.norm(vel)
        if norm > 1.0:
            vel = vel / norm
        return vel

    total = 14 * 60
    save_at = {60: 'part3_straight.png', 270: 'part3_turn.png', 600: 'part3_straight2.png', 780: 'part3_stop.png'}
    matching_video = start_video(os.path.join(VID, 'motion_matching.mp4'), (1280, 720), 60)
    for image in [title_frame('Motion matching', 'Five features: both feet, their velocity, and the future path')] * 36:
        write_frame(matching_video, image)
    for frame in range(total):
        controller.input_device.input_vel[:] = command(frame / 60.0)
        viewer.taskMgr.step()
        image = pnm_to_pil(viewer)
        write_frame(matching_video, image)
        if frame in save_at:
            image.save(os.path.join(FIG, save_at[frame]))
        if frame % 120 == 0:
            print(f'matching {frame}/{total} root {np.array2string(character.global_bone_pos, precision=2)}')
    finish_video(matching_video, os.path.join(VID, 'motion_matching.mp4'))
    viewer.destroy()


if __name__ == '__main__':
    stats, _ = main_numbers()
    feature_study(stats)
    render_all(stats)
