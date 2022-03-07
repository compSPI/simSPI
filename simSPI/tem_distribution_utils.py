from linear_simulator.params_utils import DistributionalParams
import torch




class TEMDistributionalParams(DistributionalParams):

    def get_ctf_params(self):
        if self.config.ctf:
            defocus = None
            if self.config.ctf.distribution_type == "uniform":
                min_defocus, max_defocus = self.config.ctf.distribution_params
                defocus = (
                        min_defocus
                        + (max_defocus - min_defocus)
                        * torch.zeros(self.config.batch_size)[:].uniform_()
                )

            elif self.config.ctf.distribution_type == "gaussian":
                mean, std = self.config.ctf.distribution_params
                defocus = torch.zeros(self.config.batch_size)[:].normal_(mean,std)

            else:
                raise NotImplementedError(
                    f"Distribution type: '{self.config.ctf.distribution_type}' "
                    f"has not been implemented!"
                )
            try:
                if self.config.ctf.is_numpy:
                    return defocus.detach().numpy()
            except KeyError:
                pass

        return defocus
