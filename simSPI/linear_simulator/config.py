""" Loads program configuration into a config object."""

from backports import configparser


class Config(object):
    def __init__(self, filename):
        self.load_config(filename)

    def load_config(self, filename):
        config = configparser.SafeConfigParser()
        config.read(filename)
        self.config = config

        s = "general"
        self.name = config.get(s, "name")
        self.datasetSize = config.getint(s, "datasetSize")
        self.chunks = config.getint(s, "chunks")

        s = "paths"
        self.input_volume_path = config.get(s, "input_volume_path")
        self.output_path = config.get(s, "output_path")
        self.input_starfile_path = config.get(s, "input_starfile_path")

        s = "projector"
        self.volumedomain = config.get(s, "volumedomain").lower()
        self.sidelen = config.getint(s, "sidelen")
        self.angle_distribution = config.get(s, "angle_distribution").lower()
        self.relion_invert_hand = config.getboolean(s, "relion_invert_hand")

        s = "ctf"
        self.ctf = config.getboolean(s, "ctf")
        self.changectfs = config.getboolean(s, "changectfs")
        self.ctf_size = config.getint(s, "ctf_size")
        self.valueNyquist = config.getfloat(s, "valueNyquist")
        self.bfactor = config.getfloat(s, "bfactor")
        self.pixel_size = config.getfloat(s, "pixel_size")
        self.kV = config.getfloat(s, "kV")
        self.cs = config.getfloat(s, "cs")
        self.amplitude_contrast = config.getfloat(s, "amplitude_contrast")
        self.min_defocus = config.getfloat(s, "min_defocus")
        self.max_defocus = config.getfloat(s, "max_defocus")

        s = "shift"
        self.shift = config.getboolean(s, "shift")
        self.shift_variance = config.getfloat(s, "shift_variance")
        self.shift_distribution = config.get(s, "shift_distribution").lower()

        s = "noise"
        self.noise = config.getboolean(s, "noise")
        self.noise_distribution = config.get(s, "noise_distribution").lower()
        self.noise_sigma = config.getfloat(s, "noise_sigma")
