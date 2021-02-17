import collections.abc
from os import write
from pathlib import Path
from typing import Union

import numpy as np
import soundfile
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text
import skvideo.io
from decord import VideoReader



class VideoScpReader(collections.abc.Mapping):
    """Reader class for 'video.scp'.

    Examples:
        key1 /some/path/a.mp4
        key2 /some/path/b.mp4
        key3 /some/path/c.mp4
        key4 /some/path/d.mp4
        ...

        >>> reader = VideoScpReader('wav.scp')
        >>> array = reader['key1']

    """

    def __init__(
        self,
        fname,
        tofloat=True
    ):
        assert check_argument_types()
        self.fname = fname
        self.data = read_2column_text(fname)
        self.tofloat = tofloat

    def __getitem__(self, key):
        video = self.data[key]
        #vr = VideoReader(video)
        #video = vr.get_batch(range(60)).asnumpy()
        video = skvideo.io.vread(video)
        if self.tofloat:
            video = video / 255.
        return video

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VideoScpWriter:
    """Writer class for 'video.scp'

    Examples:
        key1 /some/path/a.mp4
        key2 /some/path/b.mp4
        key3 /some/path/c.mp4
        key4 /some/path/d.mp4
        ...

        >>> writer = VideoScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="mp4",
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format

        self.data = {}

    def __setitem__(self, key: str, video):
        assert isinstance(video, np.ndarray), type(video)
        if video.ndim not in (4,):
            raise RuntimeError(f"Input signal must be 4 dimension: {video.ndim}")
        if video.dtype != np.uint8:
            video = video * 255
            video = video.astype(np.uint8)

        vpath = self.dir / f"{key}.{self.format}"
        vpath.parent.mkdir(parents=True, exist_ok=True)

        skvideo.io.vwrite(str(vpath), video)

        self.fscp.write(f"{key} {vpath}\n")

        # Store the file path
        self.data[key] = str(vpath)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


if __name__ == "__main__":
    reader = VideoScpReader('/home/chenda/workspace/develop/espnet/egs2/lrs2/enh1/data/test/video.scp')
    v = reader['6393267985458244248_00006']
    writer = VideoScpWriter('/tmp/video', '/tmp/video.scp')
    writer['testst'] = v

    writer.close()
