import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

from PauseLang_v0_7_13 import WavExporter


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),
                     'optional numpy/scipy not installed')
class WavTests(unittest.TestCase):
    def test_terminal_click_encodes_last_pause(self):
        import numpy as np
        from scipy.io import wavfile
        pauses = [.045, .150, .100]
        rate = 44100
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = str(Path(tmp)/'program.wav')
            WavExporter.export_to_wav(pauses, filename=path)
            actual_rate, audio = wavfile.read(path)
        self.assertEqual(actual_rate, rate)
        self.assertEqual(audio.dtype, np.int16)
        click = int(rate*.001)
        start = 0
        for pause in pauses:
            self.assertTrue(np.any(audio[start:start+click]))
            start += int(rate*pause)
        self.assertEqual(len(audio), start+click)
        self.assertTrue(np.any(audio[start:start+click]))

    def test_empty_stream_has_only_reference_click(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = str(Path(tmp)/'empty.wav')
            WavExporter.export_to_wav([], filename=path)
            _, audio = wavfile.read(path)
        self.assertEqual(len(audio), 44)

    def test_bad_timings_and_sample_rates(self):
        for pauses, rate in [([float('nan')],44100), ([-1],44100), ([.0001],44100),
                             ([.045],0), ([.045],3999)]:
            with self.assertRaises(ValueError):
                WavExporter.export_to_wav(pauses, sample_rate=rate)
