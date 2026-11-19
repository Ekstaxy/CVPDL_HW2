"""arbn + TENT + mean teacher (BN gamma / beta only), optionally followed by LAME.

Per batch:
  1. probe stem + layer1 of the student -> shift signal -> arbn momentum alpha_t (shared by student and teacher)
  2. teacher (EMA of the student, own Robust-BN statistics) on the clean batch -> soft targets  (no grad)
  3. student on the clean batch (statistics updated)               -> W_ENT * entropy           (TENT)
  4. student on a strongly augmented batch (stored statistics)      -> W_CONS * symmetric CE to the teacher
  5. one SGD step on the student's gamma / beta, then teacher <- MT * teacher + (1 - MT) * student
  6. prediction from PRED = "student" | "teacher" | "ensemble" (sum of logits), then LAME (k = LAME_K)

UNFREEZE = None | "layer4.2" (last residual block) | "layer4" (last stage): also train these conv weights with
their own learning rate LR_CONV (SGD, momentum 0.9, nesterov), in addition to every BN gamma / beta.

Prediction views (all computed after the statistics update, before the SGD step, eval mode, no grad: they never
change BN statistics, parameters or the RNG). Logits and features are averaged over the views before T3A / LAME.
  FLIP = True        add the horizontal flip of the image (and of every crop view)
  CROPS = (0.8,)     center crops (fraction of the side) resized back to 160, used for every image
  DARK_CROPS = (0.6,) deeper center crops used only for images whose four 12x12 corners are dark
                     (mean pixel value < DARK_THR on the 0-255 scale): an empty border around the content, as in
                     fisheye-like images. Decided per image from its own pixels; images without dark corners are
                     unaffected. Only the dark images are forwarded for these views.
FB_W > 0: feedback of the corrected answer. The final T3A + LAME probabilities of this batch (detached) become an
       extra target for the student's strongly augmented view (soft cross-entropy, weight FB_W), applied to
       FB_MASK = "both" (only images where T3A and LAME-alone both support the same label different from the
       student's) or "all" (every image). The update takes effect from the next batch.

FLIP_TGT = b > 0: the consistency target of the strongly augmented student view becomes
       (1 - b) * softmax(teacher) + b * (mean of the student's view probabilities: original + flip, before T3A /
       LAME, detached), with the symmetric CE written for a probability target. b = 0 is the original loss.
       No extra forward and no gradient through the views: gradients still come only from the augmented branch.

BANK (image bank for the consistency step; answering, TENT, T3A, LAME and the BN statistics are unchanged — bank
images only go through eval-mode forwards, so the statistics are still written once per batch by the clean batch):
  None        consistency on the 64 current images (default)
  "recent"    48 random current images + BANK_N=16 drawn uniformly from a FIFO buffer of the last BANK_SIZE=256 seen
              images, each with the teacher probabilities stored when it was seen
  "balanced"  same buffer and targets, but the 16 old images are drawn class-balanced (uniform over the teacher's
              predicted classes present in the buffer, then a random image of that class)
  "rotta"     RoTTA's CSTU memory (capacity 64, category-balanced, timeliness + uncertainty eviction); the
              consistency loss runs on the 64 memory images with targets recomputed by the current teacher and
              RoTTA's time-aware weights exp(-age) / (1 + exp(-age))

All three predictions are kept in self.last_outputs so that one run can be scored for every PRED choice.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .arbn_tent import AdaptiveRBNTent, lame
from .common import MEAN, STD, Clip, ColorJitterPro, GaussianNoise, NormalizedAug, build_ema, ema_update, entropy, get_tta_transforms, symmetric_cross_entropy
from .norm import RobustBN2d
from .rotta import CSTU, timeliness_reweighting


class AdaptiveRBNTeacher(AdaptiveRBNTent):
    MT = 0.999
    W_ENT = 1.0
    W_CONS = 1.0
    PRED = "student"
    UNFREEZE = None
    LR_CONV = 1e-3
    T3A_K = 50
    FLIP = True
    CROPS = ()
    UPSCALES = ()            # extra prediction views: the whole image resized to these sizes (e.g. (192,))
    UPSCALE_FLIP = False     # also add the horizontal flip of every upscaled view
    VIEW_AMP = False         # run the extra prediction views (flip / upscale / crops) under fp16 autocast (faster)
    DARK_CROPS = ()
    DARK_THR = 30.0
    FB_W = 0.0
    FB_MASK = "both"
    FLIP_TGT = 0.0
    BANK = None
    BANK_N = 16
    BANK_SIZE = 256
    BANK_OLD_W = 1.0         # "rotta" bank: loss weight of memory images stored before this batch (age >= 64)
                             # (with SGD this scales their effective learning rate; images of this batch keep 1)
    BANK_RESET_ALPHA = None
    AUG = "no_geom"          # strong augmentation of the consistency step: "full" (CoTTA augs) | "soft" (CoTTA soft
                             # ranges) | "no_geom" | "no_color" | "weak" (blur + noise + flip only)
    AUG_GROUPS = 1           # strong augmentation of the consistency step: split the batch into this many groups,
                             # each with its own random augmentation parameters (1 = one draw for the whole batch)
    MOM_RESET_ALPHA = None   # clear the SGD momentum buffers at a detected shift onset (alpha_t >= this value while
                             # alpha_{t-1} < it), right before that batch's optimizer.step(); gamma/beta, BN stats,
                             # teacher and T3A are kept  # "rotta" bank: empty the memory when the arbn momentum reaches this value (domain shift)

    def __init__(self, model, mt=None, w_ent=None, w_cons=None, pred=None, unfreeze="default", lr_conv=None,
                 flip=None, crops=None, dark_crops=None, fb_w=None, fb_mask=None, flip_tgt=None, bank="default",
                 bank_n=None, bank_old_w=None, mom_reset_alpha="default", aug_groups=None, aug=None, upscales=None,
                 upscale_flip=None, view_amp=None, cudnn_benchmark=None, fused_bn=None, extra_steps=0, extra_loss="both", extra_revert=False,
                 extra_stride=1, fast_bn=False, extra_tgt="teacher", shift_steps=0,
                 shift_thr=0.3, color_affine=None, shift_reset=None, shift_reset_b=1.0, lame_prev=0, unfreeze_extra=(), filter_views=(), view_avg="logit", add_pre=False, tgt_views=(), **kw):
        kw["upscales"], kw["upscale_flip"], kw["view_amp"] = upscales, upscale_flip, view_amp
        kw["cudnn_benchmark"], kw["fused_bn"] = cudnn_benchmark, fused_bn
        super().__init__(model, **{k: v for k, v in kw.items()
                                   if k not in ("upscales", "upscale_flip", "view_amp", "cudnn_benchmark", "fused_bn")})
        self.extra_steps = extra_steps   # extra SGD steps on the current batch before answering (see __call__)
        self.shift_steps, self.shift_thr = shift_steps, shift_thr  # this many more extra steps on batches whose arbn
        self.steps_hist = []                                       # momentum alpha_t >= shift_thr (a shift is under way)
        self.extra_stride = extra_stride  # extra steps use every extra_stride-th image of the batch (cheaper)
        self.extra_tgt = extra_tgt       # consistency target of the extra steps: "teacher" | "final" (provisional
                                         # T3A + LAME answer of this batch) | "mix" (average of the two)
        self.extra_loss = extra_loss     # losses of the extra steps: "both" | "ent" | "cons"
        self.extra_revert = extra_revert  # undo the extra steps after answering (the stream-long trajectory is unchanged)
        self.aug = self.AUG if aug is None else aug
        self.aug_groups = self.AUG_GROUPS if aug_groups is None else aug_groups
        self.mom_reset_alpha = self.MOM_RESET_ALPHA if mom_reset_alpha == "default" else mom_reset_alpha
        self.mom_resets = 0
        self.bank_old_w = self.BANK_OLD_W if bank_old_w is None else bank_old_w
        self.bank = self.BANK if bank == "default" else bank
        self.bank_n = self.BANK_N if bank_n is None else bank_n
        self.mem = CSTU(64, self.NUM_CLASSES) if self.bank == "rotta" else None
        self.buf_x = self.buf_p = self.buf_y = None
        self.flip_tgt = self.FLIP_TGT if flip_tgt is None else flip_tgt
        self._view_probs = None
        self.flip = self.FLIP if flip is None else flip
        self.crops = self.CROPS if crops is None else crops
        self.upscales = self.UPSCALES if kw.get("upscales") is None else kw["upscales"]
        self.upscale_flip = self.UPSCALE_FLIP if kw.get("upscale_flip") is None else kw["upscale_flip"]
        self.view_amp = self.VIEW_AMP if kw.get("view_amp") is None else kw["view_amp"]
        if kw.get("fused_bn"):  # before the teacher is copied from the student, so it inherits the flag
            for l in self.layers:
                l.fused = True
        if fast_bn:
            for l in self.layers:
                l.fast_eval = True
        if kw.get("cudnn_benchmark"):
            torch.backends.cudnn.benchmark = True  # fixed input sizes: let cuDNN pick the fastest conv algorithms
        self.dark_crops = self.DARK_CROPS if dark_crops is None else dark_crops
        self.fb_w = self.FB_W if fb_w is None else fb_w
        self.fb_mask = self.FB_MASK if fb_mask is None else fb_mask
        self.fb_count = 0
        self.unfreeze = self.UNFREEZE if unfreeze == "default" else unfreeze
        self.lr_conv = self.LR_CONV if lr_conv is None else lr_conv
        if self.unfreeze is not None:
            conv = [p for n, p in self.model.named_parameters()
                    if n.startswith(self.unfreeze + ".") and not p.requires_grad]
            for p in conv:
                p.requires_grad_(True)
            self.optimizer.add_param_group({"params": conv, "lr": self.lr_conv})
        for prefix, lr_x in unfreeze_extra:  # more frozen weights to train, each prefix with its own learning rate
            ps = [p for n, p in self.model.named_parameters() if n.startswith(prefix + ".") and not p.requires_grad]
            for p in ps:
                p.requires_grad_(True)
            self.optimizer.add_param_group({"params": ps, "lr": lr_x})
        if color_affine is not None:  # learnable 3x3 colour matrix + bias in front of conv1 (identity init), own lr
            ca = torch.nn.Conv2d(3, 3, 1, bias=True).to(next(self.model.parameters()).device)
            with torch.no_grad():
                ca.weight.copy_(torch.eye(3).view(3, 3, 1, 1)); ca.bias.zero_()
            self.model.conv1 = torch.nn.Sequential(ca, self.model.conv1)
            self.optimizer.add_param_group({"params": list(ca.parameters()), "lr": color_affine})
        self.shift_reset, self.shift_reset_b = shift_reset, shift_reset_b  # at a shift onset (alpha_t >= shift_reset
        self.view_avg = view_avg  # "logit": mean of the view logits | "logprob": mean of the view log-probabilities
        self.add_pre, self._pre = add_pre, None  # add_pre: the first (pre-update, train-mode) student forward of this
        # batch also counts as a prediction view (no extra network pass; only with predict_after)
        self.tgt_views = tuple(tgt_views)  # consistency target = mean teacher logits over the clean batch and these
        # extra views (filter_view names, "+" = also flipped; "flip" = plain flip); teacher in eval mode, fp16, no grad
        self.filter_views = tuple(filter_views)  # extra prediction views made by filtering the image (see filter_view)
        self.lame_prev, self._prev = lame_prev, None  # LAME graph: also link the last lame_prev images of the previous
        # batch (their features and final probabilities, fixed) so the first images of a batch get stream neighbours
        self.n_resets = 0  # while alpha_{t-1} < it): gamma/beta <- (1-b) * current + b * source, SGD momentum cleared
        self.mt = self.MT if mt is None else mt
        self.w_ent = self.W_ENT if w_ent is None else w_ent
        self.w_cons = self.W_CONS if w_cons is None else w_cons
        self.pred = self.PRED if pred is None else pred
        self.teacher = build_ema(self.model)
        self.t_layers = [m for m in self.teacher.modules() if isinstance(m, RobustBN2d)]
        self._t_feats = None
        self.teacher.avgpool._forward_hooks.clear()  # deepcopy also copied the student's feature hook
        self.teacher.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_t_feats", o.flatten(1)))
        self.transform = NormalizedAug(build_aug(self.aug))
        self.last_outputs = None

    def __call__(self, x):
        with torch.no_grad():
            s = self._shallow_signal(x)
            alpha = float((self.gain * s).clamp(self.alpha_min, self.alpha_max))
            for l in self.layers + self.t_layers:
                l.momentum = alpha
            self.signal_hist.append(float(s)); self.alpha_hist.append(alpha)
            self.teacher.train()
            t_out = self.teacher(x)
            t_feats = self._t_feats  # read before the target views overwrite the hook output
            t_cons = t_out
            if self.tgt_views:
                self.teacher.eval()
                acc, k = t_out.float().clone(), 1
                with torch.autocast("cuda", dtype=torch.float16):
                    for name in self.tgt_views:
                        v = torch.flip(x, dims=[3]) if name == "flip" else filter_view(x, name.rstrip("+"))
                        acc += self.teacher(v).float(); k += 1
                        if name.endswith("+"):
                            acc += self.teacher(torch.flip(v, dims=[3])).float(); k += 1
                self.teacher.train()
                t_cons = acc / k

        if self.shift_reset is not None and alpha >= self.shift_reset and \
                (len(self.alpha_hist) < 2 or self.alpha_hist[-2] < self.shift_reset):
            with torch.no_grad():
                for q, q0 in zip(self.params, self.src_params):
                    q.mul_(1 - self.shift_reset_b).add_(q0, alpha=self.shift_reset_b)
                    self.optimizer.state[q].pop("momentum_buffer", None)
            self.n_resets += 1
        self.model.train()
        with torch.enable_grad():
            with torch.set_grad_enabled(self.w_ent > 0):  # no entropy loss: this forward only updates the statistics
                s_out = self.model(x)
            s_feats = self._feats.detach()
            self._pre = (s_out.detach().float(), s_feats.float()) if self.add_pre else None
            if not self.predict_after:
                # final answer of this batch (T3A + LAME), computed before the update (it depends only on this forward)
                out, feats, cls_logits, final = self._answer(x, s_out.detach(), s_feats, t_out, t_feats)
            else:
                assert self.fb_w == 0 and self.flip_tgt == 0 and self.bank is None, "need the answer before the update"

            self.optimizer.zero_grad()
            # two separate backward passes (gradients add up): only one autograd graph is alive at a time,
            # which keeps peak VRAM well below the 10 GB limit
            if self.w_ent > 0:
                (self.w_ent * entropy(s_out).mean()).backward()
            if self.bank is not None:
                cx, ctp, cw = self._bank_batch(x, t_out)
                self.model.eval()  # augmented view: use the stored statistics, do not update them
                a_out = self.model(self.transform(cx))
                self.model.train()
                sce = -0.5 * (ctp * a_out.log_softmax(1)).sum(1) - 0.5 * (a_out.softmax(1) * ctp.clamp_min(1e-8).log()).sum(1)
                (self.w_cons * (sce * cw).mean()).backward()
            elif self.w_cons > 0 or self.fb_w > 0:
                self.model.eval()  # augmented view: use the stored statistics, do not update them
                if self.aug_groups > 1:
                    a_out = self.model(torch.cat([self.transform(c) for c in x.chunk(self.aug_groups)]))
                else:
                    a_out = self.model(self.transform(x))
                self.model.train()
                if self.flip_tgt > 0:
                    b = self.flip_tgt
                    tgt = ((1 - b) * t_out.float().softmax(1) + b * self._view_probs).detach()
                    log_tgt = tgt.clamp_min(1e-8).log()
                    sce = -0.5 * (tgt * a_out.log_softmax(1)).sum(1) - 0.5 * (a_out.softmax(1) * log_tgt).sum(1)
                    loss = self.w_cons * sce.mean()
                else:
                    loss = self.w_cons * symmetric_cross_entropy(a_out, t_cons).mean()
                if self.fb_w > 0:
                    with torch.no_grad():
                        if self.fb_mask == "all":
                            mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
                        else:
                            t3a_lab = cls_logits.argmax(1)
                            lame_only = lame(out, feats, self.lame_k).argmax(1)
                            mask = (t3a_lab == lame_only) & (t3a_lab != out.argmax(1))
                        self.fb_count += int(mask.sum())
                    fb = -(final.detach() * a_out.log_softmax(1)).sum(1) * mask.float()
                    loss = loss + self.fb_w * fb.sum() / len(x)
                loss.backward()
            if self.mom_reset_alpha is not None and self.alpha_hist[-1] >= self.mom_reset_alpha and \
                    (len(self.alpha_hist) < 2 or self.alpha_hist[-2] < self.mom_reset_alpha):
                for group in self.optimizer.param_groups:
                    for prm in group["params"]:
                        self.optimizer.state[prm].pop("momentum_buffer", None)
                self.mom_resets += 1
            self.optimizer.step()
            if self.extra_steps and self.extra_tgt != "teacher":
                with torch.no_grad():  # provisional answer from the first forward (no extra network pass)
                    z0 = self.t3a.peek(s_feats) if self.t3a is not None else s_out.detach().float()
                    z0 = lame(z0, s_feats, self.lame_k, **self.lame_kw) if self.lame_k is not None else z0.softmax(1)
                    tgt_all = z0 if self.extra_tgt == "final" else 0.5 * (z0 + t_out.float().softmax(1))
            if self.extra_steps and self.extra_revert:  # the extra steps only serve this batch's answer
                snap = [(q, q.detach().clone(), self.optimizer.state[q]["momentum_buffer"].clone())
                        for q in self.params]
            n_extra = self.extra_steps + (self.shift_steps if alpha >= self.shift_thr else 0)
            self.steps_hist.append(n_extra)
            for i in range(n_extra):  # more steps on the same batch (stored statistics, new augmentation)
                self.optimizer.zero_grad()
                self.model.eval()
                st = self.extra_stride  # every st-th image, a different offset at each step
                ex, et = (x, t_cons) if st == 1 else (x[i % st::st], t_cons[i % st::st])
                if self.w_ent > 0 and self.extra_loss in ("both", "ent"):
                    (self.w_ent * entropy(self.model(ex)).mean()).backward()
                if self.w_cons > 0 and self.extra_loss in ("both", "cons"):
                    if self.aug_groups > 1:  # same grouping as the main consistency step
                        a_out = self.model(torch.cat([self.transform(c) for c in ex.chunk(self.aug_groups)]))
                    else:
                        a_out = self.model(self.transform(ex))
                    if self.extra_tgt == "teacher":
                        sce = symmetric_cross_entropy(a_out, et)
                    else:
                        tg = tgt_all if st == 1 else tgt_all[i % st::st]
                        sce = -0.5 * (tg * a_out.log_softmax(1)).sum(1) \
                            - 0.5 * (a_out.softmax(1) * tg.clamp_min(1e-8).log()).sum(1)
                    (self.w_cons * sce.mean()).backward()
                self.model.train()
                self.optimizer.step()
        if self.predict_after:
            # learn first, then answer: the views (including the original image) go through the updated student,
            # eval mode = the statistics already updated by this batch, no second statistics update
            with torch.no_grad():
                self.model.eval()
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(self.view_amp)):
                    o = self.model(x).float()
                f = self._feats.float()
                self.model.train()
                out, feats, cls_logits, final = self._answer(x, o, f, t_out, t_feats)
        if self.extra_steps and self.extra_revert:
            with torch.no_grad():
                for q, v, mb in snap:
                    q.copy_(v); self.optimizer.state[q]["momentum_buffer"].copy_(mb)
        ema_update(self.teacher, self.model, self.mt)

        self.last_logits = out
        self.last_final = final
        return final.argmax(1)

    @torch.no_grad()
    def _answer(self, x, s_out, s_feats, t_out, t_feats):
        """Prediction views -> T3A -> LAME for this batch, with the student in its current state."""
        p_out, p_feats = self.predict_views(x, s_out, s_feats)
        self.last_outputs = {"student": (p_out, p_feats), "teacher": (t_out, t_feats),
                             "ensemble": (p_out + t_out, p_feats)}
        out, feats = self.last_outputs[self.pred]
        cls_logits = self.t3a(feats, out) if self.t3a is not None else out
        if self.lame_prev and self.lame_k is not None:
            n = self.lame_prev
            if self._prev is not None:
                pf, pp = self._prev
                z = torch.cat([pp.clamp_min(1e-8).log(), cls_logits.float().log_softmax(1)])
                final = lame(z, torch.cat([pf, feats.float()]), self.lame_k, **self.lame_kw)[len(pf):]
            else:
                final = lame(cls_logits, feats, self.lame_k, **self.lame_kw)
            self._prev = (feats.float()[-n:].clone(), final[-n:].clone())
        else:
            final = lame(cls_logits, feats, self.lame_k, **self.lame_kw) if self.lame_k is not None else cls_logits.softmax(1)
        self.last_cls = (cls_logits, feats)  # inputs of the final stage (kept for offline analysis)
        return out, feats, cls_logits, final

    @torch.no_grad()
    def _bank_batch(self, x, t_out):
        """Images, target probabilities and loss weights for the consistency step when an image bank is used."""
        tp = t_out.float().softmax(1)
        ones = lambda n: torch.ones(n, device=x.device)
        if self.bank == "rotta":
            if self.BANK_RESET_ALPHA is not None and self.alpha_hist[-1] >= self.BANK_RESET_ALPHA:
                self.mem = CSTU(64, self.NUM_CLASSES)
            ent = (-(tp * tp.clamp_min(1e-12).log()).sum(1)).tolist()
            lab = tp.argmax(1).tolist()
            for i in range(len(x)):
                self.mem.add_instance(x[i].clone(), lab[i], ent[i])
            data, ages = self.mem.get_memory()
            mx = torch.stack(data)
            self.teacher.eval()  # stored statistics only
            mtp = self.teacher(mx).float().softmax(1)
            self.teacher.train()
            w = timeliness_reweighting(ages, x.device)
            if self.bank_old_w != 1.0:  # ages are in units of the capacity (64): < 1 = added in this batch
                old = torch.tensor(ages, device=x.device) >= 1.0
                w = torch.where(old, w * self.bank_old_w, w)
            return mx, mtp, w
        n_old = 0 if self.buf_x is None else min(self.bank_n, len(self.buf_x))
        if n_old == 0:
            cx, ctp = x, tp
        else:
            keep = torch.randperm(len(x), device=x.device)[: len(x) - n_old]
            if self.bank == "recent":
                old = torch.randint(len(self.buf_x), (n_old,), device=x.device)
            else:  # balanced: uniform over predicted classes present in the buffer, then a random image of that class
                classes = self.buf_y.unique()
                pick = classes[torch.randint(len(classes), (n_old,), device=x.device)]
                old = torch.stack([(self.buf_y == c).nonzero().flatten()[
                    torch.randint(int((self.buf_y == c).sum()), (1,), device=x.device)][0] for c in pick])
            cx = torch.cat([x[keep], self.buf_x[old]])
            ctp = torch.cat([tp[keep], self.buf_p[old]])
        # remember the current batch (after sampling: only older images are replayed)
        if self.buf_x is None:
            self.buf_x, self.buf_p, self.buf_y = x.clone(), tp.clone(), tp.argmax(1)
        else:
            self.buf_x = torch.cat([self.buf_x, x])[-self.BANK_SIZE:]
            self.buf_p = torch.cat([self.buf_p, tp])[-self.BANK_SIZE:]
            self.buf_y = torch.cat([self.buf_y, tp.argmax(1)])[-self.BANK_SIZE:]
        return cx, ctp, ones(len(cx))

    @torch.no_grad()
    def predict_views(self, x, s_out, s_feats):
        """Average logits / features over the prediction views (same model state as s_out)."""
        if not (self.flip or self.crops or self.dark_crops or self.upscales or self.filter_views):
            self._view_probs = s_out.float().softmax(1)
            return s_out, s_feats
        self.model.eval()
        n = torch.ones(len(x), 1, device=x.device)
        tr = (lambda z: z.log_softmax(1)) if self.view_avg == "logprob" else (lambda z: z)
        L, Fe = tr(s_out.float()).clone(), s_feats.float().clone()
        P = s_out.float().softmax(1)

        def add(views_x, rows=None, given=None):
            nonlocal L, Fe, n, P
            if given is not None:
                o, f = given
            else:
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(self.view_amp)):
                    o = self.model(views_x).float()
                f = self._feats.float()
            o = tr(o)
            if rows is None:
                L += o; Fe += f; n += 1; P += o.softmax(1)
            else:
                L[rows] += o; Fe[rows] += f; n[rows] += 1; P[rows] += o.softmax(1)

        if self._pre is not None and self.predict_after:
            add(None, given=self._pre)
        if self.flip:
            add(torch.flip(x, dims=[3]))
        for sz in self.upscales:
            u = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False)
            add(u)
            if self.upscale_flip:
                add(torch.flip(u, dims=[3]))
        for name in self.filter_views:
            fv = filter_view(x, name.rstrip("+"))
            add(fv)
            if name.endswith("+"):  # "name+" also adds the flip of the filtered view
                add(torch.flip(fv, dims=[3]))
        for r in self.crops:
            c = center_crop(x, r)
            add(c)
            if self.flip:
                add(torch.flip(c, dims=[3]))
        if self.dark_crops:
            raw = x * STD.to(x.device) + MEAN.to(x.device)
            k = 12
            corner = torch.stack([raw[:, :, :k, :k], raw[:, :, :k, -k:], raw[:, :, -k:, :k],
                                  raw[:, :, -k:, -k:]]).mean(dim=(0, 2, 3, 4)) * 255
            rows = (corner < self.DARK_THR).nonzero().flatten()
            if len(rows):
                for r in self.dark_crops:
                    c = center_crop(x[rows], r)
                    add(c, rows)
                    if self.flip:
                        add(torch.flip(c, dims=[3]), rows)
        self.model.train()
        self._view_probs = P / n
        return L / n, Fe / n


def build_aug(mode, size=160):
    """Strong augmentation for the consistency step (on [0,1] images, wrapped by NormalizedAug)."""
    if "+" in mode:  # e.g. "soft+corr": the base augmentation followed by one random synthetic corruption
        base, extra = mode.split("+", 1)
        return T.Compose([build_aug(base, size), RandomCorruption(extra.split(","))])
    if mode == "full":
        return get_tta_transforms()
    if mode == "soft":
        return get_tta_transforms(soft=True)
    color = ColorJitterPro([0.6, 1.4], [0.7, 1.3], [0.5, 1.5], [-0.06, 0.06], [0.7, 1.3])
    geom = [T.Pad(size // 2, padding_mode="edge"),
            T.RandomAffine(degrees=[-15, 15], translate=(1 / 16, 1 / 16), scale=(0.9, 1.1),
                           interpolation=T.InterpolationMode.BILINEAR, fill=0), T.CenterCrop(size)]
    tail = [T.GaussianBlur(5, sigma=[0.001, 0.5]), T.RandomHorizontalFlip(0.5), GaussianNoise(0.005), Clip()]
    if mode == "no_geom":
        return T.Compose([Clip(), color, *tail])
    if mode == "no_color":
        return T.Compose([Clip(), *geom, *tail])
    if mode == "weak":
        return T.Compose([Clip(), *tail])
    raise ValueError(mode)


def center_crop(x, r):
    s = int(round(x.shape[-1] * r)); o = (x.shape[-1] - s) // 2
    return F.interpolate(x[:, :, o:o + s, o:o + s], size=x.shape[-2:], mode="bilinear", align_corners=False)


def filter_view(x, name):
    """Filtered copy of a normalized batch (prediction view): median3 | blur | vblur | hblur | down | gray | up192."""
    if name == "median3":
        p = F.pad(x, (1, 1, 1, 1), mode="reflect")
        u = p.unfold(2, 3, 1).unfold(3, 3, 1)
        return u.reshape(*x.shape, 9).median(-1).values
    if name in ("blur", "vblur", "hblur"):
        k = torch.tensor([1.0, 2.0, 1.0], device=x.device) / 4
        k2 = {"blur": k[:, None] * k[None, :], "vblur": k[:, None].expand(3, 3) * torch.tensor([0, 1.0, 0], device=x.device)[None, :],
              "hblur": k[None, :].expand(3, 3) * torch.tensor([0, 1.0, 0], device=x.device)[:, None]}[name]
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k2.view(1, 1, 3, 3).repeat(3, 1, 1, 1), groups=3)
    if name == "down":
        s = x.shape[-1]
        d = F.interpolate(x, size=(s * 3 // 4, s * 3 // 4), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(d, size=(s, s), mode="bilinear", align_corners=False)
    if name == "gray":
        raw = x * STD.to(x.device) + MEAN.to(x.device)
        l = (0.299 * raw[:, 0:1] + 0.587 * raw[:, 1:2] + 0.114 * raw[:, 2:3]).expand_as(raw)
        return (l - MEAN.to(x.device)) / STD.to(x.device)
    if name.startswith("up"):  # upNNN: whole image resized to NNN x NNN
        n = int(name[2:])
        return F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
    raise ValueError(name)


class RandomCorruption:
    """One randomly chosen synthetic corruption per call (whole batch, [0,1] images), with probability P.
    ops: "corr" = all of noise / lines / blur / posterize / pixelate / gray, or a list of these names."""
    P = 0.8
    ALL = ("noise", "lines", "blur", "posterize", "pixelate", "gray")

    def __init__(self, ops):
        self.ops = self.ALL if ops == ["corr"] else tuple(ops)

    def __call__(self, x):
        if float(torch.rand(())) > self.P:
            return x
        op = self.ops[int(torch.randint(len(self.ops), ()))]
        u = lambda a, b: a + (b - a) * float(torch.rand(()))
        if op == "noise":
            x = x + torch.randn_like(x) * u(0.02, 0.08)
        elif op == "lines":
            k = int(torch.randint(3, 9, ())); a = u(0.3, 0.6)
            m = torch.zeros(x.shape[-2], x.shape[-1], device=x.device)
            if float(torch.rand(())) < 0.5:
                m[int(torch.randint(k, ()))::k, :] = 1
            else:
                m[:, int(torch.randint(k, ()))::k] = 1
            x = x * (1 - a * m) + a * m * float(torch.rand(()))
        elif op == "blur":
            x = T.GaussianBlur(5, sigma=u(0.5, 1.5))(x)
        elif op == "posterize":
            L = int(torch.randint(4, 9, ()))
            x = torch.round(x.clamp(0, 1) * (L - 1)) / (L - 1)
        elif op == "pixelate":
            f = int(torch.randint(2, 4, ())); s = x.shape[-1]
            x = F.interpolate(F.avg_pool2d(x, f, ceil_mode=True), size=(s, s), mode="nearest")
        elif op == "gray":
            x = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
        return x.clamp(0, 1)
