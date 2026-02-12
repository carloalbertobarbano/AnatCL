"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 29/10/24
"""
import torch
import torch.nn as nn
import anatcl.models as models


WEIGHTS_URLS = {
    'anatcl-g3': {
        'resnet18': {
            'global': [
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_global_lambd1.0_fold0_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_global_lambd1.0_fold1_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_global_lambd1.0_fold2_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_global_lambd1.0_fold3_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_global_lambd1.0_fold4_splits5_trial0/weights.pth'
            ],

            'local': [
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_local_lambd1.0_fold0_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_local_lambd1.0_fold1_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_local_lambd1.0_fold2_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_local_lambd1.0_fold3_splits5_trial0/weights.pth',
                'https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/0_obhn64_resnet18_amp_yaware_adam_tf_none_lr0.0001_step_step10_rate0.9_wd5e-05_bsz32_trainall_False_temp0.1_views2_kernel_rbf_sigma2.0_alpha1.0_anat_desikan_normTrue_subsetTrue_local_lambd1.0_fold4_splits5_trial0/weights.pth'
            ]
        }
    },

    'anatcl-g3-old': {
        'resnet18': {
            'local': [
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold0.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold1.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold2.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold3.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold4.pth",
            ],
            'global': [
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold0.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold1.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold2.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold3.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold4.pth",
            ]
        }
    }
}


class AnatCL(nn.Module):
    def __init__(self, model="resnet18", descriptor="global", fold=0, use_head=False,
                 pretrained=True):
        super().__init__()

        self.model = model
        self.descriptor = descriptor
        self.fold = fold
        self.use_head = use_head

        self.backbone = models.SupConResNet(model, feat_dim=128, use_head=use_head)

        # Download weights from url
        weights_url = WEIGHTS_URLS['anatcl-g3'][model][descriptor][fold]
        if pretrained:
            print("Downloading weights from", weights_url)
            checkpoint = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(checkpoint['model'])

    def forward(self, x):
        return self.backbone(x)