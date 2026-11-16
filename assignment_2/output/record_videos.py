'''Record the two submission videos at real speed (60 fps).

Run from assignment_2:  python output/record_videos.py
A Panda3D window opens while frames are captured. Viewer/viewer.py is used
unmodified, so the window is on-screen (offscreen buffers trip its
requestProperties call).
'''
import os
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings('ignore')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'audio-library-name null')
loadPrcFileData('', 'sync-video false')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda3d.core import ClockObject, Filename, LVector3, PNMImage, WindowProperties, getModelPath

from file_io import BVHMotion
from task1_motion_editing import concatenate_two_motions
from Viewer.viewer import SimpleViewer
from Viewer.controller import Controller
from task2_motion_matching import CharacterController, InteractiveUpdate

FPS = 60
W, H = 1280, 720
VID = os.path.join(ROOT, 'output', 'videos')
TMP = os.path.join(ROOT, 'output', '_frame.png')
os.makedirs(VID, exist_ok=True)


class MouseStub:
    def is_button_down(self, *args, **kwargs):
        return False


def open_viewer():
    getModelPath().prependDirectory(Filename.from_os_specific(ROOT))
    viewer = SimpleViewer()
    props = WindowProperties()
    props.setSize(W, H)
    viewer.win.requestProperties(props)
    viewer.setFrameRateMeter(False)
    viewer.mouseWatcherNode = MouseStub()
    globalClock.setMode(ClockObject.MNonRealTime)
    globalClock.setDt(1.0 / FPS)
    for _ in range(5):
        viewer.graphicsEngine.renderFrame()
    return viewer


def capture(viewer):
    viewer.graphicsEngine.renderFrame()
    image = PNMImage()
    viewer.win.getScreenshot(image)
    image.write(Filename.from_os_specific(TMP))
    img = Image.open(TMP).convert('RGB')
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return img


def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def label(img, title, lines=()):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    height = 56 + 34 * len(lines)
    draw.rectangle((0, 0, 900, height), fill=(20, 24, 28))
    draw.text((18, 10), title, fill=(245, 245, 242), font=font(30))
    for k, line in enumerate(lines):
        colour = line[1] if isinstance(line, tuple) else (200, 205, 200)
        text = line[0] if isinstance(line, tuple) else line
        draw.text((18, 52 + 34 * k), text, fill=colour, font=font(24))
    return img


def title_card(text, subtitle):
    img = Image.new('RGB', (W, H), (28, 32, 36))
    draw = ImageDraw.Draw(img)
    draw.text((80, 280), text, fill=(245, 245, 242), font=font(54))
    draw.text((80, 370), subtitle, fill=(190, 198, 190), font=font(28))
    return img


class Video:
    def __init__(self, path):
        self.path = path
        self.log = open(path + '.log', 'w')
        self.proc = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '20', '-movflags', '+faststart', path,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.frames = 0

    def write(self, img, repeat=1):
        data = img.convert('RGB').tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(data)
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait()
        self.log.close()
        if code != 0:
            raise RuntimeError('ffmpeg failed for ' + self.path)
        os.remove(self.path + '.log')
        print(f'{self.path}: {self.frames} frames, {self.frames / FPS:.1f} s')


def follow(viewer, root):
    viewer.cameractrl.center = LVector3(float(root[0]), 0.95, float(root[2]))
    viewer.cameractrl.position = LVector3(float(root[0]) + 2.3, 1.85, float(root[2]) - 3.15)
    viewer.cameractrl.look()


def render_clip(viewer, motion):
    trans, ori = motion.batch_forward_kinematics()
    frames = []
    for f in range(len(trans)):
        for i, name in enumerate(motion.joint_name):
            viewer.set_joint_position_orientation(name, trans[f, i], ori[f, i])
        follow(viewer, trans[f, 0])
        frames.append(capture(viewer))
    return frames


def load_pair():
    walk = BVHMotion('data/motion_walking.bvh')
    run = BVHMotion('data/motion_running.bvh')
    run.adjust_joint_name(walk.joint_name)
    return walk, run


def concatenation_video(viewer):
    between = 8
    walk, run = load_pair()
    direct = walk.raw_copy()
    direct.local_joint_positions = np.concatenate([walk.local_joint_positions[:40], run.local_joint_positions[0:]])
    direct.local_joint_rotations = np.concatenate([walk.local_joint_rotations[:40], run.local_joint_rotations[0:]])
    direct_phases = [(0, 40, 'walk (motion 1)'), (40, len(direct.local_joint_positions), 'run (motion 2)')]

    clips = [('1. Direct splice (do_interp=False)', 'Walk frame 40 -> run frame 0, no search, no blend',
              direct, direct_phases)]
    for method, title, sub in (
        ('interpolation', '2. Interpolation, 8 in-between frames',
         'Search window +-20, matched walk 25 -> run 18, lerp + slerp, Hermite root'),
        ('inertialization', '3. Inertialization, half-life 8 frames',
         'Run plays from the matched frame, the pose offset decays'),
    ):
        walk, run = load_pair()
        res = concatenate_two_motions(walk, run, 40, 0, between, method=method)
        # Matched cut: res = walk[:real_i] + between + run[real_j:]. Redo the
        # same window search (rows: walk 20..59, cols: run 0..19) for real_i.
        n_between = between if method == 'interpolation' else 0
        w, r = load_pair()
        a = w.local_joint_rotations[20:60].copy(); a[a[..., 3] < 0] *= -1
        b = r.local_joint_rotations[0:20].copy(); b[b[..., 3] < 0] *= -1
        sim = np.linalg.norm(a.reshape(len(a), -1)[:, None] - b.reshape(len(b), -1)[None], axis=-1)
        real_i = 20 + int(np.argmin(sim)) // sim.shape[1]
        assert len(res.local_joint_positions) == real_i + n_between + len(r.local_joint_positions) - (int(np.argmin(sim)) % sim.shape[1])
        print(method, 'cut at walk frame', real_i)
        phases = [(0, real_i, 'walk (motion 1)')]
        if n_between:
            phases.append((real_i, real_i + n_between, 'blend (in-between frames)'))
        phases.append((real_i + n_between, len(res.local_joint_positions),
                       'run (motion 2)' + (', offset decaying' if method == 'inertialization' else '')))
        clips.append((title, sub, res, phases))

    video = Video(os.path.join(VID, 'motion_concatenation.mp4'))
    video.write(title_card('Motion concatenation: walk -> run',
                           'Each version plays once at real speed (60 fps)'), FPS * 2)
    slow = None
    for title, sub, motion, phases in clips:
        frames = render_clip(viewer, motion)
        video.write(title_card(title, sub), int(FPS * 1.5))
        n = len(frames)
        for f, img in enumerate(frames):
            phase = next(p for a, b, p in phases if a <= f < b)
            colour = (255, 200, 90) if phase.startswith('blend') else (200, 205, 200)
            img = label(img, title, [(f'frame {f + 1}/{n}   {phase}', colour)])
            video.write(img, FPS // 2 if f in (0, n - 1) else 1)
        if title.startswith('2.'):
            slow = (title, frames, phases)
    title, frames, phases = slow
    video.write(title_card('Transition close-up', 'Interpolation version, 0.25x speed, frames 15-50'), int(FPS * 1.5))
    for f in range(15, min(50, len(frames))):
        phase = next(p for a, b, p in phases if a <= f < b)
        colour = (255, 200, 90) if phase.startswith('blend') else (200, 205, 200)
        img = label(frames[f], 'Close-up at 0.25x speed', [(f'frame {f + 1}   {phase}', colour)])
        video.write(img, 4)
    video.close()


def matching_video():
    viewer = open_viewer()
    controller = Controller(viewer)
    names = ['lFootPos', 'rFootPos', 'lFootVel', 'rFootVel', 'trajectoryPos2D']
    weights = [0.75, 0.75, 1.0, 1.0, 1.0]
    character = CharacterController(viewer, controller, names, weights)
    viewer.addTask(InteractiveUpdate(viewer, controller, character).update)

    # (end second, stick vector, caption)
    script = [
        (3.0, (0, 0, 1), 'forward'),
        (6.0, (1, 0, 1), 'forward + left'),
        (8.5, (0, 0, 1), 'forward'),
        (11.5, (-1, 0, 1), 'forward + right'),
        (14.0, (0, 0, 1), 'forward'),
        (17.0, (0, 0, 0), 'release stick (stop)'),
    ]

    def command(sec):
        for end, vec, cap in script:
            if sec < end:
                v = np.array(vec, dtype=float)
                n = np.linalg.norm(v)
                return (v / n if n > 1 else v), cap
        return np.zeros(3), script[-1][2]

    video = Video(os.path.join(VID, 'motion_matching.mp4'))
    video.write(title_card('Motion matching', '5 features: lFootPos, rFootPos, lFootVel, rFootVel, trajectoryPos2D'),
                FPS * 2)
    total = int(script[-1][0] * FPS)
    for f in range(total):
        vel, cap = command(f / FPS)
        controller.input_device.input_vel[:] = vel
        viewer.taskMgr.step()
        img = capture(viewer)
        video.write(label(img, 'Motion matching (scripted stick input)',
                          [f'stick: {cap}', f't = {f / FPS:4.1f} s']))
    video.close()
    viewer.destroy()


if __name__ == '__main__':
    which = sys.argv[1:] or ['concat', 'matching']
    if 'concat' in which:
        v = open_viewer()
        concatenation_video(v)
        v.destroy()
    if 'matching' in which:
        matching_video()
    if os.path.exists(TMP):
        os.remove(TMP)
