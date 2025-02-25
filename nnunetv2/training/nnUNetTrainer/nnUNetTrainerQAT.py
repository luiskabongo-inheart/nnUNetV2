from .nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerQAT(nnUNetTrainer):
    use_qat=True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1
        self.configuration_manager.network_arch_init_kwargs["norm_op_kwargs"]["track_running_stats"] = True
