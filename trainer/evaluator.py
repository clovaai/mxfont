"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import utils


def torch_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()

        return ret

    return decorated


class Evaluator:
    def __init__(self, writer):
        torch.backends.cudnn.benchmark = True
        self.writer = writer

    @torch_eval
    def comparable_val_saveimg(self, gen, loader, step, n_row, tag='val'):
        compare_batches = self.infer_fact_loader(gen, loader)
        comparable_grid = utils.make_comparable_grid(*compare_batches[::-1], nrow=n_row)
        saved_path = self.writer.add_image(tag, comparable_grid, global_step=step)

        return comparable_grid, saved_path

    @torch_eval
    def infer_fact_loader(self, gen, loader, save_dir=None):
        outs = []
        trgs = []

        for batch in loader:
            style_imgs = batch["style_imgs"].cuda()
            char_imgs = batch["source_imgs"].unsqueeze(1).cuda()

            out = gen.gen_from_style_char(style_imgs, char_imgs)
            outs.append(out.detach().cpu())
            if "trg_imgs" in batch:
                trgs.append(batch["trg_imgs"])

        outs = torch.cat(outs).float()
        ret = (outs,)
        if trgs:
            trgs = torch.cat(trgs)
            ret += (trgs,)

        return ret
