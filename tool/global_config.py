import numpy as np


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class GlobalConfig(object):
    def __init__(
        self,
        dtype="float64",
        use_gpu=True,
        max_subsequence_ratio_p=0.95,
        min_align_ratio_p=0.1,
        use_msa=True,
    ):
        self.NP_DTYPE = None
        dtype in ("float32", "float64")
        self.DTYPE = dtype
        self.AMBER_DEVICE = "CUDA" if use_gpu else "CPU"
        self.STEREO_CHEMICAL_PROPS = None
        self.MAX_SUBSEQUENCE_RATIO_P = max_subsequence_ratio_p
        self.MIN_ALIGN_RATIO_P = min_align_ratio_p
        self.DEV_MODE = False
        self.DEV_PROF_MEM_MODE = False
        self.DEV_PROF_TIME_MODE = False
        self.FASTFOLD_OPTIMIZE = False
        self.PLM_CKPT = None
        self.FLOW_ATTN = False
        self.PAIR_FACTOR=False
        self.DEV_PROF_MEM_MAIN_MODE = False
        self.ADD_BIAS_S = False
        self.ADD_BIAS_V = False

    def set_dtype(self, dtype):
        self.DTYPE = dtype
        self.NP_DTYPE = np.float64 if dtype == "float64" else np.float32

    def set_amber_device(self, use_gpu=True):
        self.AMBER_DEVICE = "CUDA" if use_gpu else "CPU"

    def set_stereo_chemical_props(self, stereo_chemical_props):
        self.STEREO_CHEMICAL_PROPS = stereo_chemical_props

    def set_subseq_align_ratio(
        self, max_subsequence_ratio_p=0.95, min_align_ratio_p=0.1
    ):
        """
        Antibody:
            The threshold is detemined by our antibody benchmark. Because the antibody sequence are very similar,
            we want to specify a strict threshold for capture the real templates with similar structure.
            max_subsequence_ratio = 1.0
            min_align_ratio = 0.97
        Antigen:
            max_subsequence_ratio = 0.95
            min_align_ratio = 0.1
        """
        self.MAX_SUBSEQUENCE_RATIO_P = max_subsequence_ratio_p
        self.MIN_ALIGN_RATIO_P = min_align_ratio_p

    def set_dev_mode(self, dev_mode):
        self.DEV_MODE = dev_mode

    def set_dev_prof_mem_mode(self, dev_prof_mem_mode):
        self.DEV_PROF_MEM_MODE = dev_prof_mem_mode
    def set_dev_prof_mem_main_mode(self, prof_memory_main):
        self.DEV_PROF_MEM_MAIN_MODE = prof_memory_main
        
    def set_dev_prof_time_mode(self, dev_prof_time_mode):
        self.DEV_PROF_TIME_MODE = dev_prof_time_mode

    def set_reproduce(self, is_reproduce, reproduce_seed):
        self.IS_REPRODUCE = is_reproduce
        self.REPRODUCE_SEED = reproduce_seed

    def set_fastfold_optimize(self, use_fastfold_optimize):
        self.FASTFOLD_OPTIMIZE = use_fastfold_optimize

    def set_plm_ckpt(self, ckpt_path):
        self.PLM_CKPT = ckpt_path
    
    def set_flow_attn(self, flow_attn):
        self.FLOW_ATTN = flow_attn
        
    def set_performer_attn(self, performer_attn):
        self.PER_ATTN = performer_attn
        
    def set_pair_factor(self, pair_factor):
        self.PAIR_FACTOR = pair_factor
    
    def set_add_bias_v(self,add_bias_v):
        self.ADD_BIAS_V = add_bias_v

    def set_add_bias_s(self,add_bias_s):
        self.ADD_BIAS_S = add_bias_s
