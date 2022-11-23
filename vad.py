import collections
import contextlib
from genericpath import exists
import sys
import wave
import os
import glob
import argparse
from tqdm import tqdm
import shutil

import webrtcvad


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield (b''.join([f.bytes for f in voiced_frames]), frame.timestamp)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def get_args():
    parser = argparse.ArgumentParser(description="Preprocessing the audio files from datasets.")
    parser.add_argument("--level", type=int, required=False, default=2, help="agressiveness about filtering")
    parser.add_argument("--data_dir", type=str, required=True, default='/data/', help="Dataset root path")
    parser.add_argument("--save_dir", type=str, required=True, default='/data/vad/', help="Trimmed file save path")
    parser.add_argument("--mode", type=str, required=True, help="train or test mode")
    args = parser.parse_args()
    return args


def main(args):

    vad = webrtcvad.Vad(int(args.level)) # Create Vad Object

    data_path = os.path.join(args.data_dir, args.mode)
    save_path = os.path.join(args.save_dir, args.mode)

    if args.mode == "train":
        spk_ids = sorted([f.split('/')[-1] for f in glob.glob(data_path + '/*') if os.path.isdir(f)])

        for spk_id in tqdm(spk_ids):
            utts = glob.glob(f'{data_path}/{spk_id}/*.wav')

            save_dir = os.path.join(save_path, spk_id)
            os.makedirs(save_dir, exist_ok=True)

            for utt in utts:
                audio, sample_rate = read_wave(utt)
                frames = frame_generator(30, audio, sample_rate) # Create frame_generator Object
                frames = list(frames)
                segments = vad_collector(sample_rate, 30, 300, vad, frames)
                try:
                    next(segments)
                except StopIteration as exc:
                    shutil.copyfile(utt, file_path)
                else:
                    for segment in segments:
                        utt_name = utt.split('/')[-1]
                        file_path = os.path.join(save_dir,utt_name)
                        print(' Writing %s' % (file_path,) + '\n')
                        try:
                            write_wave(file_path, segment, sample_rate)
                        except:
                            shutil.copyfile(utt, file_path)
    else:
        utts = glob.glob(f'{data_path}/*.wav')
        os.makedirs(save_path, exist_ok=True)
        print(f'utts:{len(utts)}')
        for utt in tqdm(utts):
            audio, sample_rate = read_wave(utt)
            frames = frame_generator(30, audio, sample_rate) # Create frame_generator Object
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)
            try:
                next(segments)
            except StopIteration as exc:
                print(exc.value)
                shutil.copyfile(utt, file_path)
            else:
                for segment in segments:
                    utt_name = utt.split('/')[-1]
                    file_path = os.path.join(save_path,utt_name)
                    print(' Writing %s' % (file_path,) + '\n')
                    try:
                        write_wave(file_path, segment, sample_rate)
                    except:
                        shutil.copyfile(utt, file_path)

    # utt = '/home/ubuntu/speaker_recognition/data/test/51trdoSev7B.wav'
    # file_path = '/home/ubuntu/speaker_recognition/ujinne/data/51trdoSev7B.wav'
    # audio, sample_rate = read_wave(utt)
    # frames = frame_generator(30, audio, sample_rate) # Create frame_generator Object
    # frames = list(frames)
    # # print(len(frames))
    # segments = vad_collector(sample_rate, 30, 300, vad, frames)
    # try:
    #     next(segments)
    # except StopIteration as exc:
    #     print(exc.value)
    #     shutil.copyfile(utt, file_path)
    # else:
    #     for segment in segments:
    #         print(type(segment))
    #         utt_name = utt.split('/')[-1]
    #         # file_path = os.path.join(save_path,utt_name)
    #         print(' Writing %s' % (file_path,) + '\n')
    #         try:
    #             write_wave(file_path, segment, sample_rate)
    #         except:
    #             shutil.copyfile(utt, file_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
