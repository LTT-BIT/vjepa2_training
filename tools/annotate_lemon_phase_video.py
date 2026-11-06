#!/usr/bin/env python3
"""Burn per-head-frame phase predictions into the original rosbag camera video."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAGES = ('左臂到位前', '左臂到位 → 开始舀取', '开始舀取 → 抬起',
          '抬起 → 倒入完成', '倒入完成后')
V3_STAGES = ('建立左手支撑', '对准铲子', '铲取柠檬片', '移向杯口上方', '倒入杯子', '任务完成')
COLORS = ('#68c8dd', '#f1c75b', '#93cb75', '#eaa06c', '#cd9cdc', '#6fdfac')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path)
    source.add_argument('--video', type=Path)
    parser.add_argument('--title', help='Original episode name for a converted video')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter-root', type=Path, default=Path('/home/descfly/Convert_data'))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline='')))
    if not rows:
        raise ValueError('Prediction CSV is empty')
    meta_path = args.csv.with_suffix('.meta.json')
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    stages = V3_STAGES if metadata.get('label_semantics') == 'five_readiness_boundaries_plus_task_complete_v3' else STAGES
    if {k for k in rows[0] if k.startswith('p') and k[1:].isdigit()} != {f'p{i}' for i in range(len(stages))}:
        raise ValueError('Probability columns do not match label semantics; six-state CSV requires its metadata')
    font = ImageFont.truetype(FONT, 28)
    big = ImageFont.truetype(FONT, 42)
    small = ImageFont.truetype(FONT, 21)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        if args.bag:
            import yaml
            sys.path.insert(0, str(args.converter_root / 'src'))
            from convest.sources.gello_rosbag2 import Bag
            topic = yaml.safe_load((args.converter_root / 'schemas/ur_o30_contract.yaml').read_text())['topics']['cam0']['topic']
            bag = stack.enter_context(Bag(args.bag, args.converter_root / 'schemas'))
            frames = sorted(bag.records(topic, image=True), key=lambda item: item[0])
            if len(frames) != len(rows):
                raise ValueError(f'{len(frames)} bag frames != {len(rows)} CSV rows')
            for index, ((timestamp, _), row) in enumerate(zip(frames, rows)):
                if int(row['frame_index']) != index or int(row['header_ns']) != timestamp:
                    raise ValueError(f'Head image and prediction differ at frame {index}')
            references = [reference for _, reference in frames]
            read_frame = bag.image
            fps, title = 10., args.bag.name
        else:
            import cv2
            cap = cv2.VideoCapture(str(args.video))
            stack.callback(cap.release)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f'Cannot read video: {args.video}')
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise ValueError('Video and CSV frame counts differ')
            for index, row in enumerate(rows):
                time = float(row['time_s'])
                if not np.isfinite(time) or abs(time - index / fps) > 1e-4:
                    raise ValueError(f'Video and prediction time differ at frame {index}')
            def read_frame(reference):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode video frame {reference}')
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            references = range(len(rows))
            title = args.video.stem
        for index, row in enumerate(rows):
            if row.get('valid', '1') == '1':
                probabilities = np.array([float(row[f'p{phase}']) for phase in range(len(stages))])
                if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or abs(probabilities.sum() - 1) > 1e-3 or int(row['phase']) != probabilities.argmax():
                    raise ValueError(f'Invalid probabilities at frame {index}')
            elif row['valid'] != '0' or row['phase']:
                raise ValueError(f'Invalid status at frame {index}')

        first = read_frame(references[0])
        camera_height, width = first.shape[:2]
        if width < 1000 or camera_height < 600:
            raise ValueError('Head image is too small for this annotation layout')
        height = camera_height + 280
        temporary = args.output.with_suffix('.tmp.mp4')
        command = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                   '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                   str(temporary)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for index, (reference, row) in enumerate(zip(references, rows)):
                image = Image.new('RGBA', (width, height), (15, 22, 26, 255))
                image.paste(Image.fromarray(first if index == 0 else read_frame(reference)).convert('RGBA'), (0, 64))
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle((0, 0, width, 64), fill=(15, 22, 26, 255))
                draw.rectangle((0, height - 215, width, height), fill=(15, 22, 26, 225))
                draw.text((28, 13), f'{args.title or title}  ·  头部相机', font=font, fill='white')
                draw.text((width - 345, 19), f"{float(row['time_s']):06.2f}s   帧 {index + 1:03d}/{len(rows)}",
                          font=small, fill='#dddddd')
                valid = row.get('valid', '1') == '1'
                phase = int(row['phase']) if valid else None
                color = COLORS[phase] if valid else '#f1c75b'
                draw.text((30, height - 198), '模型预测阶段' if valid else '当前帧无有效预测',
                          font=small, fill='#cccccc')
                draw.text((28, height - 164), f'阶段 {phase}  {stages[phase]}' if valid else '腕部/关节对齐数据不足',
                          font=big if valid else font, fill=color)
                if valid:
                    confidence = float(row[f'p{phase}'])
                    draw.text((30, height - 96), f'当前阶段概率 {confidence:.1%}', font=font, fill='white')
                draw.text((30, height - 46), '五时间戳：阶段 4 = 倒入，阶段 5 = 完成；模型预测' if stages == V3_STAGES else '旧标注语义：阶段 4 = 倒入完成后；请人工确认',
                          font=small, fill='#dddddd')
                for phase_index in range(len(stages)):
                    y = height - 196 + phase_index * (30 if len(stages) == 6 else 36)
                    probability = float(row[f'p{phase_index}']) if valid else 0
                    draw.text((width - 370, y - 2), f'p{phase_index}', font=small,
                              fill=COLORS[phase_index] if phase == phase_index else '#cccccc')
                    draw.rectangle((width - 319, y + 7, width - 111, y + 21), fill='#465052')
                    draw.rectangle((width - 319, y + 7, width - 319 + round(208 * probability), y + 21),
                                   fill=COLORS[phase_index])
                    draw.text((width - 94, y - 2), f'{probability:5.1%}' if valid else '--',
                              font=small, fill='white')
                image.alpha_composite(overlay)
                process.stdin.write(image.convert('RGB').tobytes())
        finally:
            process.stdin.close()
            status = process.wait()
        if status:
            raise RuntimeError(f'ffmpeg exited with status {status}')
        temporary.replace(args.output)
    print(f'{len(rows)} annotated frames -> {args.output}')


if __name__ == '__main__':
    main()
