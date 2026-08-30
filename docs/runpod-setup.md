# RunPod runbook — finishing Phase 3

The one-session procedure that takes Phase 3 from "pipeline built, zero
results" to all four ablation arms reported. Written to be followed
top-to-bottom without improvising.

**Why we left Kaggle.** Nothing was ever wrong with the training loop. Five
runs died with their *session* — four on Kaggle, one on Colab — every one
with a healthy loss curve and no weights at the end (`docs/journal.md`,
2026-07-23 and 2026-08-10). Two things fix that, and neither is a code
change: a **persistent volume** that outlives the pod, and running training
under **nohup** so it is not bound to a browser tab. Everything below is
built around those two.

The secondary win is hardware. Kaggle's T4 is Turing — no bf16 tensor
cores — which is why this project runs fp16 there, and fp16 is what
produced the silent gradient-underflow bug on 2026-07-19. Any Ampere or
Ada card has bf16, which carries fp32's exponent range and cannot underflow
that way. `train_lora.resolve_precision()` detects this and picks the right
mode; nothing needs to be set by hand.

---

## 0. Paying, from India

**RunPod does not accept UPI.** Neither does Vast.ai or Lambda. Payment is
Stripe (Visa / Mastercard / Amex) or crypto with KYC. For our purposes:

- **Use an Indian Visa or Mastercard — credit or debit.** These work on
  RunPod. HDFC / ICICI / SBI / Axis cards are all fine.
- **Enable international transactions first.** Since March 2020 the RBI
  requires every newly issued or reissued card to ship with international
  usage *disabled*. This is the single most common reason the payment
  fails. Turn it on in your bank's app (usually Cards → Manage →
  International / Overseas usage), and set the limit above your top-up.
- **RuPay will not work** — Stripe does not support it.
- **Avoid prepaid / forex cards** — RunPod requires a $100 minimum deposit
  on those, which defeats the point.
- If the card still declines, the fallbacks are a different issuer's card,
  crypto (needs KYC — overkill for $10), or an INR-billing Indian provider
  (E2E Networks, Yotta) as a last resort.

**Load $10.** That covers the training run, the evaluation, and one full
re-run if something goes wrong. Billing is per second, so nothing is lost
by finishing early.

---

## 1. Create the network volume *first*

This is the step that makes the run survivable, so it comes before the pod.

1. RunPod console → **Storage** → **New Network Volume**.
2. Size: **50 GB**. Cost: $0.07/GB/month ≈ **$3.50/month**, prorated
   hourly — pennies for one night, but see step 8, it keeps billing until
   you delete it.
3. Pick a datacenter that has RTX 4090 availability.

Two things to know:

- **Network volumes are Secure Cloud only.** Community Cloud ($0.34/hr) is
  cheaper but cannot attach one. Take Secure Cloud at $0.69/hr — the
  difference over a ~5 hour run is about $1.70, against a failure mode
  that has now cost five runs. That is the whole reason we're here.
- **The volume is datacenter-bound**, and attaching it restricts you to
  GPUs in that datacenter. If no 4090 is free there, make the volume in a
  different datacenter rather than giving up the volume.

## 2. Deploy the pod

1. **Pods** → **Deploy** → **Secure Cloud**.
2. GPU: **RTX 4090 (24 GB)**, 1×. (A40 48 GB at $0.44/hr also works and is
   cheaper per hour but slower; A100 80 GB at $1.39/hr finishes in roughly
   half the time for about the same total. The 4090 is the sweet spot.)
3. Template: **RunPod PyTorch** (any recent CUDA 12.x PyTorch image).
4. Under storage, **attach the network volume from step 1**. It mounts at
   `/workspace` and replaces the default disk.
5. Container disk: 30 GB is plenty — the base model download (~4 GB) and
   pip packages live there.
6. Deploy, then open **Connect → Web Terminal** (or SSH).

Sanity check before anything else:

```bash
nvidia-smi                      # confirm an RTX 4090 is attached
python -c "import torch; print(torch.cuda.is_bf16_supported())"   # must print True
df -h /workspace                # must show the ~50GB network volume
```

If `is_bf16_supported()` prints `False`, you are on a Turing card, not an
Ada one. Stop and redeploy — `TrainingArguments(bf16=True)` will refuse to
construct anyway, so this fails loudly rather than silently, but there is
no reason to pay for the wrong card.

## 3. Get the code onto the pod

```bash
cd /workspace
git clone https://github.com/rudranaresh0201/computer-use.git
cd computer-use
pip install -e ".[training]"
python -c "import computeruse; print('ok')"
```

`pyproject.toml` pins `transformers==5.0.0` deliberately — a newer release
changed Qwen2-VL's forward pass to require an `mm_token_type_ids` input our
`collate_fn` doesn't produce, which killed the 2026-07-23 Colab run on its
first step. Don't upgrade it.

If `pip install -e ".[training]"` fails on `pywinauto` (a Windows-only
dependency of the base package, not of training), fall back to:

```bash
pip install -e . --no-deps
pip install transformers==5.0.0 peft jinja2 torchvision pyyaml
export PYTHONPATH=/workspace/computer-use/src
```

## 4. Transfer the dataset

`runpodctl` is preinstalled on every pod. Install it locally on Windows
once (see runpod.io/console/user/settings → CLI), then from your **local**
machine:

```powershell
cd "C:\Users\Rudra\Desktop\New folder\project computer use"
runpodctl send data\gui_grounding\gui_grounding_upload.zip
runpodctl send runs\uia_arm_results.json
```

Each prints a one-time code. On the **pod**, for each:

```bash
mkdir -p /workspace/data /workspace/runs
cd /workspace/data && runpodctl receive <code>          # the zip
cd /workspace/runs && runpodctl receive <code>          # the UIA results
unzip -q /workspace/data/gui_grounding_upload.zip -d /workspace/data/gui_grounding
ls /workspace/data/gui_grounding                        # images/ labels.jsonl splits/
```

Two notes on what is being transferred:

- The zip is 150 MB and contains `images/`, `labels.jsonl`, and `splits/`.
  It does **not** contain `apps.yaml` — that is tracked source, not
  generated data, so it arrives with the git clone instead.
  `runpod_eval.py` resolves it from the repo automatically.
- `runs/uia_arm_results.json` is the UIA-only arm, already computed on
  Windows (`scripts/export_uia_arm.py`). That arm needs `pywinauto` and
  cannot run on Linux at all, so it travels as data. Without it the two
  model arms still run, but `uia_only` and `hybrid` are skipped.

## 5. Launch training — detached

**This is the step that has failed five times. Do not run it in the
foreground.**

```bash
cd /workspace/computer-use
mkdir -p /workspace/runs
nohup python scripts/runpod_train.py \
    --dataset-root /workspace/data/gui_grounding \
    --output-dir /workspace/runs/lora_grounder \
    > /workspace/runs/train.log 2>&1 &

echo $!            # note the PID
tail -f /workspace/runs/train.log
```

`Ctrl-C` on the `tail` detaches your view; it does not stop training. You
can close the browser, shut the laptop, lose wifi — the process survives,
and its checkpoints are on the network volume, which survives even pod
termination.

What the log should show in the first two minutes:

```
gpu:       NVIDIA GeForce RTX 4090
precision: bf16
batch:     2 x 2 accum = 4 effective
trainable params: ... || all params: ... || trainable%: 0.8290
no checkpoint found -- starting fresh from step 0
```

If `trainable%` is not ~0.829, stop — the adapter did not attach correctly
and you would be training nothing.

**The run:** 1,572 train examples at an effective batch of 4 = **393 steps
per epoch, 1,179 steps for the full 3 epochs.** Expect 15–20 s/step, so
**5–6.5 hours, about $3.40–$4.50.** Checkpoints land every 100 steps, and
relaunching the exact same command resumes from the last one.

**If it OOMs** (unlikely on 24 GB, but screenshots make long visual token
sequences): rerun with `--batch-size 1 --grad-accum 4`. The product must
stay at 4 — that is the effective batch every hyperparameter, `lr=1e-4`
especially, was chosen against. The script refuses to start if it isn't.

**Optional, while training runs** — bank the baseline arms now, in a second
terminal, so a later failure can't cost you everything:

```bash
python scripts/runpod_eval.py --skip-fine-tuned \
    --dataset-root /workspace/data/gui_grounding \
    --uia-results /workspace/runs/uia_arm_results.json \
    --out /workspace/runs/baselines_report.json
```

## 6. Evaluate — all four arms

Once `train.log` prints `adapter saved to .../final`:

```bash
python scripts/runpod_eval.py \
    --dataset-root /workspace/data/gui_grounding \
    --adapter /workspace/runs/lora_grounder/final \
    --uia-results /workspace/runs/uia_arm_results.json \
    --out /workspace/runs/ablation_report.json
```

This scores 1,591 examples (dev + test_same_app + test_held_out_app) under
each arm and writes the report sliced three ways — by arm, by arm × app
richness, by arm × split — which is exactly what H1/H2/H3 are read off.
Roughly 20–40 minutes for both model arms.

## 7. Pull the results down

From the pod:

```bash
cd /workspace/runs
tar czf phase3_results.tar.gz ablation_report.json train.log lora_grounder/final
runpodctl send phase3_results.tar.gz
```

Locally, `runpodctl receive <code>`. The adapter is a few tens of MB — it
is LoRA, not a full model.

## 8. Shut down — actually check this

1. **Terminate the pod.** A merely *stopped* pod still bills volume disk at
   $0.20/GB/month.
2. **Delete the network volume** once the results are safely downloaded.
   It bills until deleted, and RunPod explicitly says it is not designed
   for long-term storage — data can be removed if your balance expires.
3. Confirm the balance stopped moving on the billing page.

---

## Reading the result honestly

Before the numbers turn into a claim, two things are already known and
should be stated up front rather than discovered by a reader:

**The UIA arm is a ceiling, not a competitor.** It scores **99.8%**, and
resolves a candidate on 1,588 of 1,591 examples. That is by construction,
and `eval/uia_only.py`'s own docstring says so: the dataset contains only
elements UIA could already see, so the correct element is *always* in the
candidate pool. It measures "can text matching pick the right element from
a list that definitely contains it" — not "can UIA find click targets in
the wild." Because hybrid falls back to the grounder only on the 3 examples
UIA couldn't resolve, **hybrid will come out ≈ UIA-only**, and H2's
"hybrid beats both" will be true in a way that is close to trivial.

So the real result of this run is **H1: fine-tuned vs zero-shot**, and
**H3: does the fine-tuned arm beat zero-shot on the four held-out apps.**
Those are honest, and they are what the write-up should lead with.

**Scale, stated plainly.** 200 screenshots, 1,572 training rows. The
comparable published work (GroundCUA, June 2026) uses 55,568 screenshots
with human verification. This is a small, honest replication of a known
recipe on a platform the literature doesn't cover, plus a failure analysis
of accessibility-tree auto-labeling — not a competitor to those numbers.
An H1 that holds and an H3 that doesn't is a perfectly good, reportable
outcome.
