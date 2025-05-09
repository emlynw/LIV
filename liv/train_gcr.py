import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from PIL import Image
import hydra
import torch
import torchvision.transforms as T
import pandas as pd
import time

# --- LIV / GCR imports -------------------------------------------------------
from liv import load_liv                      # optional snapshot load
from liv.gcr_trainer import GCRTrainer        # joint VIP + GCR loss
from liv.utils.data_loaders import GCROfflineBuffer
from liv.utils import utils
from liv.utils.logger import Logger
from liv.utils.plotter import plot_reward_curves

# -----------------------------------------------------------------------------
# helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def make_network(cfg):
    model = hydra.utils.instantiate(cfg)
    # if torch.cuda.device_count() > 1:       # keep the wrapper only when needed
    #     model = torch.nn.DataParallel(model)
    return model.to(cfg.device)


# -----------------------------------------------------------------------------
# Workspace -------------------------------------------------------------------
# -----------------------------------------------------------------------------

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.logging = self.cfg.logging
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        if self.logging:
            self._setup_logger()

        # ------------------------- dataloader ---------------------------------
        if not cfg.eval:
            print("Creating GCR‑Offline dataloader …")
            train_iters = GCROfflineBuffer(
                datapath=self.cfg.datapath_train,
                beta=self.cfg.data.beta,
                num_workers=self.cfg.num_workers,
                doaug=self.cfg.doaug,
                alpha=self.cfg.alpha,
            )
            print("success-episodes:", len(train_iters.manifest_s))
            print("neg-buffer len  :", len(train_iters.neg_buf))
            # keep a reference for the trainer
            self.neg_buf = train_iters.neg_buf
            self.train_loader = iter(
                torch.utils.data.DataLoader(
                    train_iters,
                    batch_size=self.cfg.batch_size,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True,
                )
            )

        # ---------------------------- model -----------------------------------
        print("Initialising LIV model …")
        self.model = make_network(cfg.agent)
        print(f"model: {self.model}")
        self.timer = utils.Timer()
        self._global_step = 0

        # ------------------------- snapshot load ------------------------------
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self._load_snapshot(cfg.load_snap)

        # ------------------------- trainer ------------------------------------
        if not cfg.eval:
            self.trainer = hydra.utils.instantiate(
                self.cfg.trainer,
                neg_buf=self.neg_buf,   # pass static negative buffer
            )

    # ---------------------------------------------------------------------
    # setup / utils --------------------------------------------------------
    # ---------------------------------------------------------------------

    def _setup_logger(self):
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    # ---------------------------------------------------------------------
    # main loops -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def train(self):
        until = utils.Until(self.cfg.train_steps, 1)
        every_eval = utils.Every(self.cfg.eval_freq, 1)

        print("Begin offline GCR training …")
        while until(self.global_step):
            # ---------------- evaluation / snapshot ----------------------
            if every_eval(self.global_step):
                if self.cfg.agent.grad_text:          # only if the text tower is present
                    self._generate_reward_curves()
                else:
                    print("⚠  Text tower disabled – skipping language-distance plots.")
                self._save_snapshot()

            # ------------------- single gradient step --------------------
            t0 = time.time()
            batch = next(self.train_loader)
            t1 = time.time()
            metrics, _ = self.trainer.update(self.model, batch, self.global_step)
            t2 = time.time()

            # ------------------- logging ---------------------------------
            if self.logging:
                self.logger.log_metrics(metrics, self.global_step, ty='train')

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f"Sample time {t1 - t0:.3f}s | Update time {t2 - t1:.3f}s")

            self._global_step += 1

    # ------------------------------------------------------------------
    # snapshot helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _save_snapshot(self):
        path      = self.work_dir / f"snapshot_{self.global_step}.pt"
        full_path = self.work_dir / "snapshot.pt"

        model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        state     = {
            "liv":        model_ref.state_dict(),      # <-- fixed
            "optimizer":  model_ref.encoder_opt.state_dict(),
            "global_step": self._global_step,
        }
        torch.save(state, path)
        torch.save(state, full_path)

    def _load_snapshot(self, path):
        if path == "liv":
            self.model = load_liv()  # loads official LIV weights
            return
        payload = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(payload["liv"])
        try:
            self._global_step = payload["global_step"]
        except KeyError:
            print("[snapshot] warning: no global_step found; starting from 0")

    # ------------------------------------------------------------------
    # evaluation visual -------------------------------------------------
    # ------------------------------------------------------------------

    def _generate_reward_curves(self):
        self.model.eval()
        os.makedirs(self.work_dir / "reward_curves", exist_ok=True)
        transform = T.Compose([T.ToTensor()])

        # choose manifest + task list as in original script
        manifest = pd.read_csv(Path(self.cfg.datapath_train) / "manifest_success.csv")
        tasks = manifest["text"].unique()

        def load_video(row):
            vid = Path(row["directory"])
            imgs = []
            for i in range(row["num_frames"]):
                for ext in (".png", ".jpg"):
                    f = vid / f"{i}{ext}"
                    if f.exists():
                        imgs.append(transform(Image.open(f)))
                        break
            return torch.stack(imgs)

        fig_name = self.work_dir / "reward_curves" / f"{self._global_step}_{self.cfg.dataset}.png"
        plot_reward_curves(manifest, tasks, load_video, self.model, fig_name, animated=self.cfg.animate)
        self.model.train()


# -----------------------------------------------------------------------------
# entry point -----------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path='cfgs', config_name='config_gcr')
def main(cfg):
    root_dir = Path.cwd()
    ws = Workspace(cfg)

    # auto‑resume if snapshot exists
    snap_file = root_dir / 'snapshot.pt'
    if snap_file.exists() and not cfg.eval:
        print(f"[resume] loading {snap_file}")
        ws._load_snapshot(snap_file)

    if cfg.eval:
        if cfg.agent.grad_text:
            ws._generate_reward_curves()
    else:
        ws.train()


if __name__ == '__main__':
    main()