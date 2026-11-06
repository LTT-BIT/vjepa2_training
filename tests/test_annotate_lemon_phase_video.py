"""Run with .venv-phase/bin/python tools/test_annotate_lemon_phase_video.py."""
from pathlib import Path
import subprocess
import json
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def test_converted_video_alignment():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        video, csv, output = (root / name for name in ('head.mp4', 'phase.csv', 'annotated.mp4'))
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 800))
        assert writer.isOpened()
        for value in (40, 180):
            writer.write(np.full((800, 1280, 3), value, dtype=np.uint8))
        writer.release()
        command = [sys.executable, str(Path(__file__).with_name('annotate_lemon_phase_video.py')),
                   '--video', str(video), '--csv', str(csv), '--output', str(output), '--title', 'episode-test']
        header = 'time_s,phase,p0,p1,p2,p3,p4\n'
        csv.write_text(header + '0,0,1,0,0,0,0\n')
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode and 'frame counts differ' in result.stderr
        csv.write_text(header + '0,0,1,0,0,0,0\n0.3,1,0,1,0,0,0\n')
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode and 'time differ' in result.stderr
        csv.write_text(header + '0,0,1,0,0,0,0\n0.1,1,0,1,0,0,0\n')
        subprocess.run(command, check=True, capture_output=True)
        cap = cv2.VideoCapture(str(output))
        try:
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
            assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1080
            for value in (40, 180):
                ok, image = cap.read()
                assert ok and abs(float(image[464, 640].mean()) - value) < 12
        finally:
            cap.release()
        csv.write_text('time_s,phase,p0,p1,p2,p3,p4,p5\n0,4,0,0,0,0,1,0\n0.1,5,0,0,0,0,0,1\n')
        csv.with_suffix('.meta.json').write_text(json.dumps({'label_semantics': 'five_readiness_boundaries_plus_task_complete_v3'}))
        subprocess.run(command, check=True, capture_output=True)
        cap = cv2.VideoCapture(str(output))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
        cap.release()


if __name__ == '__main__':
    test_converted_video_alignment()
    print('Converted-video alignment and frame-order checks passed')
