# GenAssoc RUN-PREPARATION DOCTRINE (2026-07-06)

**Author:** Monzia Moodie
**Purpose:** Every GenAssoc run (1-17) has failed at *start* on an environment problem
that was predictable and preventable. This document + `scripts/vm_bootstrap_run.sh`
make that entire class of failure structurally impossible. It is the required reading
and required first step before any cloud run.

**The core principle, stated once:** *Everything drifts.* The cloud CLI drifts, the base
image drifts, pip dependencies drift, remote model code drifts, and this project's own
scripts go stale relative to all of the above. Drift is not an incident to react to; it
is the default state of the software universe, to be anticipated and gated against
*before* spending money on compute. The agent layer, the smokes, the probes, and this
doctrine all exist for that single reason.

---

## 1. THE ORDERED LAUNCH SEQUENCE (do these in this order, every time)

Run start is a pipeline of gates. Each gate is cheap; each prevents an expensive failure
later. Do NOT skip ahead — the ordering is the point (a data pull before the stack is
pinned wastes the transfer; a run before models load wastes the box).

**Local, before touching a VM (free, do these first):**
0. **Confirm CLI currency.** `vastai --version` >= the version the runbook was written for.
   If newer, re-verify `create instance` / `destroy instance` / `search offers` syntax
   against docs — *scripts written weeks ago are stale by default* (this bit Run 17: the
   `cpu_ram` GB/MB change and the `destroy instance` subcommand both moved).
1. **Pin the NT revision in code** (see §3) — the one permanent fix that removes the
   reproducibility hole. Do this once; it protects every future run.
2. **Push or archive the repo at the intended HEAD.** If GitHub push is blocked by a large
   blob in history, use `git archive HEAD` -> 30 MB tar -> scp (proven in Run 17).
3. **Provision network-first.** Filter offers by `Net_down >= 1000` (Mbps) AND price cap,
   THEN by value. A slow-network host stalls the image pull (Run 17: 35 min in `loading`,
   destroyed). Fast host booted in ~1 min.

**On the fresh VM, in order:**
4. **`bash scripts/vm_bootstrap_run.sh`** — the environment gate (system tools, rclone,
   pinned Python stack, model-load proof, GPU). REFUSES to proceed unless green. This is
   the new mandatory first step; it encodes every §2 failure mode below.
5. **`bash /workspace/vm_pull_run17_data.sh`** — data staging: pulls every REQUIRED file
   from Drive to the launcher's exact paths, hard-verifies presence+size, fail-loud.
6. **`MIN_DISK_GB=80 bash scripts/Run_Preflight_VM.sh`** — hardware/dep preflight.
7. **`bash scripts/launch_run17_baseline.sh`** — the real run.
8. **Postflight per `POSTFLIGHT_RUN17_PROTOCOL.md`**, then teardown
   (`vastai destroy instance <id>`), then confirm gone (`vastai show instances`).

Gates 4-6 are all fail-loud and must be GREEN before the next. No exceptions — the whole
reason runs fail at start is skipping or assuming a gate.

---

## 2. FAILURE-MODE CATALOG (every start-of-run failure and its permanent guard)

Each row is a real failure this project hit. The bootstrap script's phase (A-E) that now
prevents it is named. This table is the institutional memory; add to it after every run.

| # | Failure (observed) | Run | Root cause | Permanent guard |
|---|---|---|---|---|
| 1 | `vastai search offers` returned zero rows | 17 | `cpu_ram` is GB in CLI, not MB; `cpu_ram>=64000` asked 64 TB | Search helper uses GB; §1.0 CLI-currency check |
| 2 | `vastai destroy <id>` would fail | 17 | CLI moved to `destroy instance <id>`; no `-y` | Corrected teardown script; §1.0 |
| 3 | VM stuck `loading` 35 min, destroyed | 17 | Picked host by value, ignored `Net_down` (361 Mbps) | §1.3 network-first provisioning |
| 4 | GitHub push rejected | 17 | 1.95 GB AlphaFold blob in git history | `git archive HEAD` bypass (§1.2); history purge deferred |
| 5 | `rclone: command not found` | 17 | Base image has no rclone | Bootstrap Phase B installs it |
| 6 | rclone install failed: no unzip | 17 | Base image has no unzip/7z/busybox | Phase A installs unzip; Phase B uses python-unzip fallback |
| 7 | Data paths mismatched Drive layout | 17 | rnaseq nested+renamed; R13 under r13/annotations/; constraint path | `vm_pull_run17_data.sh` maps every path from a verified manifest |
| 8 | Preflight FAIL: torch_geometric/networkx/imodelsx absent | 17 | Not in base image | Phase C installs them |
| 9 | pandas silently upgraded 2.3->3.0 | 17 | `imodelsx` dragged pandas 3.0 over the pinned stack | Phase C pins to lock, re-pins after imodelsx |
| 10 | transformers upgraded 5.8->5.13 | 17 | Same imodelsx drift | Phase C pins transformers to lock |
| 11 | NT load ImportError `find_pruneable_heads_and_indices` | 17 | NT `from_pretrained` has NO `revision=`; pulls head-of-main remote code that imports a symbol transformers removed. Local cache masks it. | §3 revision pin (code) + Phase D model-load proof (gate) |
| 12 | Disk floor near-miss after 60 GB FinnGen | 17 | 150 GB default floor vs real ~20 GB need | `MIN_DISK_GB=80` documented override |
| 13 | Empty-file attachments wasted round-trips | 17 | Large VM logs arrived empty in chat | Paste short output directly; `tee` + scp-back + size-check before attaching |
| 14 | `show instances` deprecation warning | 17 (teardown) | CLI moved to `show instances-v1` — drift happened MID-SESSION | §1.0 currency check; the live proof that drift is continuous, not a one-time fix |
| 15 | NT "works locally" via stale modules cache | 17 | An untracked `modeling_esm.py` in `~/.cache/.../modules/` silently satisfied a dep; not in the trusted base's inventory | Vendor NT modeling file into VCS (§3); clean-cache validation |

**The meta-lesson (the one that matters most):** items 1-2 and 5-6 and 8-11 are all the
same *class* — "a thing that lives outside our repo changed since we last looked." They
were individually surprising only because each was treated as a surprise. The bootstrap
treats the whole class as expected: it re-derives truth from the environment (lock file,
live remote listing, actual model load) every time, so staleness is caught by construction
rather than discovered mid-run.

---

## 3. THE NT REVISION PIN (the one permanent code fix — do this first, locally)

**The bug:** `src/genomic_variant_classifier/data/genomic_lm.py` loads the Nucleotide
Transformer with `AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)`
— no `revision=`. `trust_remote_code` fetches the CURRENT `modeling_esm.py` from the NT
repo. That head-of-main file imports `find_pruneable_heads_and_indices` from
`transformers.pytorch_utils`, which transformers REMOVED (absent in the pinned 5.8.0). So
NT breaks on any fresh HF cache. The developer's local machine has an OLDER cached
`modeling_esm.py` that predates the breaking import — which is why it works locally and
nowhere else. Classic unpinned-dependency reproducibility hole.

**CRITICAL CORRECTION (2026-07-06, after teardown):** the revision pin ALONE does NOT
fix this. Evidence: the local working cache is revision
`f34324c6fde36a4f635f0f1f06cac5d25acd6798`, and the failed VM pulled *that exact same
revision* and still broke. So the problem is NOT "VM got a newer revision" — it is that
this revision's remote `modeling_esm.py` imports `find_pruneable_heads_and_indices`, which
is absent from the pinned `transformers==5.8.0`. The developer's local machine only works
because it has a STALE copy of `modeling_esm.py` cached under `~/.cache/huggingface/modules/`
(a *different* cache dir from `hub/snapshots/`) that predates the breaking import line. That
stale module is the hidden trusted-base artifact making local "work." Pinning `revision=`
to the same hash reproduces the same failure on a clean box.

**The real fix — pick ONE, validate locally against a clean modules cache before trusting:**
1. **Vendor NT's modeling file (most robust, recommended).** Copy the WORKING
   `modeling_esm.py` (+ `esm_config.py`) from the local
   `~/.cache/huggingface/modules/transformers_modules/InstaDeepAI/.../` into the repo, and
   load NT from the local vendored path instead of `trust_remote_code=True` off the Hub.
   This removes the Hub-remote-code dependency entirely — the file is version-controlled,
   diffable, and cannot drift. Pin `revision=` too, for the weights.
2. **Pin transformers to a version whose API matches the remote code.** Find the
   transformers version that still exports `find_pruneable_heads_and_indices` (a 4.x line),
   and pin BOTH transformers and NT revision to a mutually-compatible pair. Trade-off: your
   lock currently says `transformers==5.8.0` for other reasons (ESM-2, tokenizers); changing
   it needs a full re-validation of the ESM-2 path. Verify ESM-2 still loads.
3. **Patch NT's remote code at the pinned revision to not need the removed symbol** — a real
   vendored edit (option 1 in disguise), NOT a runtime monkeypatch of transformers.

**Validation that actually proves the fix (must clear a CLEAN modules cache):**
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/InstaDeepAI*   # clear the stale masker
python -c "from transformers import AutoModelForMaskedLM; AutoModelForMaskedLM.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-100m-multi-species', trust_remote_code=True, revision='f34324c6fde36a4f635f0f1f06cac5d25acd6798')"
```
If that succeeds on a CLEARED modules cache, the fix is real; if it reproduces the ImportError,
your local "working" state was an artifact of the stale cache and you need option 1 or 2. This
clean-cache test is the ground-truth check — do NOT trust a load that reused the old module.

Until a real fix lands, the bootstrap's Phase D FAILS LOUD on NT (it loads on a fresh box's
clean cache, which is exactly the condition that exposes this), so no run proceeds on broken NT.

**The deeper lesson (feeds the drift doctrine):** "works on my machine" was literally a STALE
CACHED FILE outside version control silently satisfying a dependency. This is the canonical
drift trap — a trusted-base artifact you didn't know was in the trusted base. The fix is to
pull it INTO version control (vendor it), shrinking the untracked trusted base to zero for NT.

**Apply the same discipline to any other `trust_remote_code=True` load** (audit for it):
an unpinned remote-code model is a latent version bomb.

---

## 4. STANDING PRE-PROVISION CHECKLIST (paste-and-tick before spending on a box)

```
[ ] vastai --version >= runbook version; create/destroy/search syntax re-verified if newer
[ ] NT revision pinned in genomic_lm.py (or NT_REVISION hash in hand for the bootstrap)
[ ] requirements.lock is current and in the repo tree being shipped
[ ] repo archived at intended HEAD (git archive) OR pushed; HEAD sha noted
[ ] offer chosen network-FIRST: Net_down >= 1000 Mbps, price < cap, reliability >= 0.98
[ ] rclone.conf ready to scp (token risk accepted for short-lived box; destroy after)
[ ] plan: bootstrap -> data pull -> preflight -> run -> postflight -> destroy
[ ] MIN_DISK_GB override known (80) if FinnGen (60 GB) is in the tree
```

If any box is left running, remember `Max_Days` on the host — some offer ~1 day and
auto-terminate. Destroy explicitly when done; never rely on idle.

---

## 5. WHY THIS IS WORTH THE DISCIPLINE

The cost of a start-of-run failure is not just the wasted VM dollars — it is the wasted
*preparation* time, the destroyed run that produced no scientific data, and the erosion of
trust that the pipeline will run when it matters. The bootstrap+doctrine turn a ~2-hour
error-laden provisioning slog into a fixed, ~10-minute, verified, fail-loud sequence whose
every step either passes visibly or stops the line with a named remedy. That is the
difference between "hope it runs" and "know it runs before the meter starts."
