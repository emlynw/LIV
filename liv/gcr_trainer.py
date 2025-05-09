# liv/gcr_trainer.py  (replaces the earlier version)
import torch
import torch.nn.functional as F
from liv.trainer import Trainer            # just to borrow the loss utilities
from liv.gcr_buffer import GCRNegBuffer

class GCRTrainer(Trainer):
    def __init__(self, neg_buf: GCRNegBuffer,
                 w1: float = 1.0, w2: float = 1.0):
        super().__init__()
        self.neg_buf, self.w1, self.w2 = neg_buf, w1, w2

    # ---------- new code ----------
    def _contrastive_loss(self, sim_pos, sim_neg):
        # Eq. (3) from the paper
        return -self.w1 * sim_pos.mean() + self.w2 * sim_neg.mean()

    # ---------- one‑pass update ----------
    def update(self, model, batch, step, eval=False):
        metrics = {}
        if eval:
            model.eval()
        else:
            model.train()

        # === unpack and encode exactly as in Trainer.update ===
        b_im, b_reward, b_lang = batch
        b_im = b_im.cuda()
        bs, stack, _, H, W = b_im.shape
        enc = model(b_im.reshape(bs*stack, 3, H, W), modality="vision")
        enc = enc.view(bs, stack, -1)

        e0, eg = enc[:,0], enc[:,1]          # initial & goal
        es0, es1 = enc[:,2], enc[:,3]        # (o_t , o_{t+1})

        # -------- VIP loss (unchanged) --------
        vip_loss = self.compute_vip_loss(model, e0, es0, es1, eg,
                                         b_reward, model.module.num_negatives)
        metrics['vip_loss'] = vip_loss.item()

        # -------- GCR contrastive loss --------
        # positive goal pair (eg, eg′)
        pos_idx = torch.randint(1, stack-1, (1,))
        eg_p = enc[:, pos_idx[0]]

        # negative goals sampled online
        if len(self.neg_buf) > 0:
            g_neg_imgs = torch.stack(self.neg_buf.sample(bs)).cuda()
            e_gneg = model(g_neg_imgs, modality="vision")
        else:                                 # warm‑up
            e_gneg = eg.detach()

        sim_pos = model.module.sim(eg, eg_p)
        sim_neg = model.module.sim(eg, e_gneg)
        gcr_loss = self._contrastive_loss(sim_pos, sim_neg)
        metrics['gcr_loss'] = gcr_loss.item()

        # -------- total loss & single optimizer step --------
        full_loss = (model.module.visionweight * vip_loss
                     + model.module.visionweight * gcr_loss)
        metrics['full_loss'] = full_loss.item()

        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()

        return metrics, None
