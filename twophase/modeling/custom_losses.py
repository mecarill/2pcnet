import torch
from torch.nn import KLDivLoss

class ConsistencyLosses:
    def __init__(self):
        self.kldivloss = KLDivLoss(reduction="none", log_target=False)

    def losses(self,student_roi,teacher_roi):
        loss = {}
        class_scores_student = []
        class_scores_teacher = []
        for s_roi, t_roi in zip (student_roi, teacher_roi):
            class_scores_student.append(s_roi.full_scores) #[:,:-1])
            class_scores_teacher.append(t_roi.full_scores) #[:,:-1])
        class_scores_student=torch.cat(class_scores_student,axis=0)
        class_scores_teacher=torch.cat(class_scores_teacher,axis=0)

        # Weighted KL Divergence
        weights = class_scores_teacher.max(axis=1).values
        kl_loss = self.kldivloss(torch.log(class_scores_student),class_scores_teacher)
        kl_loss = kl_loss.mean(axis=1)*weights
        kl_loss = torch.mean(kl_loss)

        loss['loss_cls_pseudo'] = kl_loss

        return loss
    