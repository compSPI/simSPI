from CifFile import ReadCif
import bisect
from ioSPI.ioSPI import datasets
from pathlib import Path


def upload_dataset_from_files(token: str, data_file: str, metadata_file: str, data_node_guid: str = "24htr") -> bool:
    """Upload particle stack data and metadata as labelled datasets to OSF.io.
        Parameters
        ----------
        TODO: update docstring.
        data_file : str
            File path to .h5 file containing datasets.
        metadata_file : str
            File path to .star file containing relevant meta data.
        Returns
        -------
        bool
            True if all uploads successful, False otherwise.
        """

    # Structure
    #   Data Node (datasets) -> Structure Node (eg: 4v6x) -> Dataset Node (eg: 4v6x_randomrots) -> data file + meta data file

    # Workflow
    # get structure label eg. 4v6x
    # get dataset label eg. 4v6x_random_rotations
    # check if node with structure label exists in osf
    # if not, create it
    # create a new node for dataset under structure node
    # get tags for dataset
    # upload files with tags

    osf = datasets.OSFUpload(token, data_node_guid)

    with open(metadata_file, 'r') as file:
        parsed_metadata = ReadCif(file, grammar = 'STAR2')

    structure_name = Path(parsed_metadata['particle']['_pdb_file']).stem
    dataset_name = Path(parsed_metadata['simulation']['_log_file']).stem

    structure_guid = osf.read_structure_guid(structure_name)

    if not structure_guid:
        structure_guid = osf.write_child_node(osf.data_node_guid, structure_name)

    # get tags

    required_tags = ["pixel_size_um", "noise_type", "noise_level", "ctf_distribution"]
    retrieved_tags = get_tags_from_parsed_metadata(parsed_metadata, required_tags)

    dataset_guid = osf.write_child_node(structure_guid, dataset_name, retrieved_tags)

    upload_file_paths = [data_file, metadata_file]
    return osf.write_files(dataset_guid, upload_file_paths)


def get_tags_from_parsed_metadata(parsed_metadata, tags):
    """Generate tags using parsed metadata from TEMSim.
    Parameters
    ----------
    meta_data

    Returns
    -------

    """
    # from https://github.com/compSPI/ioSPI/issues/20
    #
    #   pixel_size_ang [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] : does it make sense to have bins or discrete values?. Use _detector_pixel_size_um
    #   particle_size [small(<128px),mid(128-256px),large(<512px)] : does not exist in metadata. found in fov functions.#TODO: add to meta_data
    #   noise_type [gaussian,none]: if signal to noise key exists -> gaussian
    #   noise_level [what values/bins do we want to deal with?]: if noise is not none, use snr ratio.
    #   ctf_distribution [uniform,gaussian,none] : from ctf -> _distribution_type
    #   rotation_distribution["uniform_on_sphere"] : keep it this for now #TODO: add to meta_data
    #   n_particles["<1000,1000-2500,2500-5000,5000+"] : no of tilts * no of particles in src. bins are arbitrily  chosen.TODO: add no of particles from fov to meta_data

    # TODO: implemented this.

    tags_metadata_map = {
        "pixel_size_um": "_detector_pixel_size_um",
        "noise_type": ("_signal_to_noise", lambda stn_ratio: "gaussian" if bool(stn_ratio) else "none"),
        "noise_level": "_signal_to_noise",
        "ctf_distribution": "_distribution_type"
    }

    tag_dict = {}
    for tag_label in tags:

        metadata_map = tags_metadata_map.get(tag_label)

        if not metadata_map:
            raise KeyError(f"Tag '{tag_label}' not supported.")

        if type(metadata_map) is tuple:
            metadata_value = parsed_metadata.get_all(metadata_map[0])
            metadata_value = metadata_value[0] if bool(metadata_value) else None
            metadata_transform = metadata_map[1]
            tag_value = metadata_transform(metadata_value)
        else:
            tag_value = parsed_metadata.get_all(metadata_map)[0]

        tag_dict[tag_label] = tag_value

    return [f"{label} : {value}" for label, value in tag_dict.items()]


if __name__ == "__main__":
    with open('./4v6x_randomrot.star', 'r') as file:
        parsed_metadata = ReadCif(file, grammar = 'STAR2')

    print(get_tags_from_parsed_metadata(parsed_metadata, ["ctf_distribution", "noise_type"]))
    # upload_dataset_from_files('as','./4v6x_randomrot.star', './4v6x_randomrot.star')
