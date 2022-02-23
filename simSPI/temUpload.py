from CifFile import ReadCif

from ioSPI.ioSPI import datasets
from pathlib import Path


def upload_dataset_from_files(token: str, data_file: str, metadata_file: str, data_node_guid: str = "24htr") -> bool:
    """Upload particle stack data and metadata as labelled datasets to OSF.io.
        Parameters
        ----------
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

    #get tags
    tags = []

    dataset_guid = osf.write_child_node(structure_guid, dataset_name,tags)

    upload_file_paths = [data_file,metadata_file]
    return osf.write_files(dataset_guid, upload_file_paths)






def get_tags_from_parsed_metadata(meta_data:CifFile):

    # from https://github.com/compSPI/ioSPI/issues/20
    #
    #   pixel_size_ang [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] : does it make sense to have bins or discrete values?. Use _detector_pixel_size_um
    #   particle_size [small(<128px),mid(128-256px),large(<512px)] : does not exist in metadata. found in fov functions.#TODO: add to meta_data
    #   noise_type [gaussian,none]: if signal to noise key exists -> gaussian
    #   noise_level_db [what values/bins do we want to deal with?]: if noise is not none, use snr ratio.
    #   ctf_distribution [uniform,gaussian,none] : from ctf -> _distribution_type
    #   rotation_distribution["uniform_on_sphere"] : keep it this for now
    #   n_particles["<1000,1000-2500,2500-5000,5000+"] : no of tilts * no of particles in src. bins are arbitrily  chosen.TODO: add no of particles from fov to meta_data

    #TODO: implemented this.





if __name__ == "__main__":
    upload_dataset_from_files('./4v6x_randomrot.star', './4v6x_randomrot.star')

