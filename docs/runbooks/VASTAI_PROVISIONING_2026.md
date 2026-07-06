# VAST.AI PROVISIONING REFERENCE — CLI 1.2.0 (2026-07-06)

**Author:** Monzia Moodie
**Applies to:** `vastai` Python CLI **1.2.0** (released 2026-07-02). Verify with `vastai --version`.
**Supersedes:** the CLI-command specifics in older runbooks/scripts written for `vastai` 1.0.x.
**Compute target:** single RTX 4090 (or fallback) for the Run-17 GPU run. **Price cap: < $0.80/hr.**

This doc is the authoritative, current command reference. It exists because the CLI changed between
1.0.13 and 1.2.0 in ways that silently broke the old commands — most importantly a units change that
made `search offers` return ZERO rows. Every command below was checked against the 2026 Vast.ai docs,
the vast-cli source, and the 1.2.0 PyPI release.

---

## 0. THE BUG THAT RETURNED ZERO OFFERS (root cause, documented so it never recurs)

The query used on 2026-07-06 returned a header and **zero rows**:
```
vastai search offers "gpu_name=RTX_4090 ... cpu_ram>=64000 ..." -o "dlperf_usd-"
```
**Cause:** in the `vastai` CLI, `cpu_ram` is in **GIGABYTES**, not megabytes. `cpu_ram>=64000`
therefore requested 64,000 GB (≈64 TB) of system RAM — which no machine has — zeroing the whole set.
(The REST API expresses `cpu_ram` in MB, e.g. `65536` = 64 GB; the CLI auto-converts and expects GB.
The same GB-in-CLI / MB-in-API split applies to `gpu_ram`.)

**Fix:** use `cpu_ram>=64` (for 64 GB). All other fields, booleans, operators, and the `-o` sort key
were valid. Over-filtering (too many stacked constraints on thin 4090 supply) is the secondary risk.

---

## 1. FIELD REFERENCE — `vastai search offers` (units matter)

| Purpose | Field | Unit / notes | Example |
|---|---|---|---|
| GPU model | `gpu_name` | string, underscores, no quotes on value | `gpu_name=RTX_4090` |
| GPU count | `num_gpus` | int | `num_gpus=1` |
| Per-GPU VRAM | `gpu_ram` | **GB** in CLI (MB in REST) | `gpu_ram>=24` |
| System RAM | `cpu_ram` | **GB** in CLI (MB in REST) ← THE BUG | `cpu_ram>=64` |
| Disk | `disk_space` | GB | `disk_space>=200` |
| Reliability | `reliability` | 0-1 | `reliability>=0.98` |
| Verified host | `verified` | bool | `verified=true` |
| Rentable now | `rentable` | bool | `rentable=true` |
| Direct/SSH ports | `direct_port_count` | int | `direct_port_count>=1` |
| Total price | `dph_total` | $/hr (GPU+disk) | `dph_total<0.80` |
| Sort key | `-o` / `--order` | trailing `-` = descending | `-o 'dph_total'` |

Operators: `>= > <= < == != in notin`. Wrap the WHOLE query in quotes so the shell doesn't eat `>`/`<`.
`in` lists widen supply: `gpu_name in [RTX_4090,RTX_5090,RTX_3090]`.

---

## 2. CORRECTED SEARCH QUERIES (Run-17 preflight floors: VRAM>=20, RAM>=50, disk>=150)

```bash
# Cheapest single RTX 4090 under $0.80/hr, price-sorted:
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true dph_total<0.80' -o 'dph_total'

# The original intent, corrected (64 GB RAM / 200 GB disk / verified / direct / reliable):
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true direct_port_count>=1 disk_space>=200 cpu_ram>=64 reliability>=0.98 dph_total<0.80' -o 'dph_total'

# Widen supply across comparable GPUs (recommended if 4090 stock is thin):
vastai search offers 'gpu_name in [RTX_4090,RTX_5090,RTX_3090] num_gpus=1 rentable=true disk_space>=200 cpu_ram>=64 dph_total<0.80' -o 'dph_total'
```
**If a query returns zero rows, relax constraints IN THIS ORDER** (test after each removal):
`disk_space>=200` → `direct_port_count>=1` → `reliability>=0.98` → `verified=true`. If a minimally
filtered query (`gpu_name=RTX_4090 num_gpus=1 rentable=true`) STILL returns nothing, it is genuine
supply scarcity, not syntax — switch GPU model.

---

## 3. GPU FALLBACKS UNDER $0.80/hr (all clear the Run-17 floors: VRAM>=20, RAM>=50, disk>=150)

- **RTX 5090 (32 GB GDDR7)** — the strongest fallback; MORE VRAM than a 4090, ~$0.53/hr on-demand
  (mid-2026). First choice if 4090 supply is thin.
- **RTX 4090 (24 GB)** — primary target; on-demand frequently < $0.80/hr, spot as low as ~$0.15-0.31/hr.
- **RTX 3090 (24 GB)** — abundant, usually well under $0.40/hr; solid 24 GB value.
- **A100 40/80 GB** — occasionally dips to ~$0.67/hr on high-reliability hosts but usually > $0.80;
  treat sub-cap A100 as opportunistic.
- **RTX A5000/A6000, L40S, RTX 6000 Ada** — pro cards that intermittently appear sub-$0.80 on spot.

Prices fluctuate constantly across Vast's 40+ datacenters — always confirm live with a
`dph_total`-sorted query before renting. The figures here are mid-2026 snapshots.

---

## 4. FULL WORKFLOW — CLI 1.2.0, exact current syntax

```bash
# 0. one-time
pip install -U vastai                      # 1.2.0+; verify: vastai --version
vastai set api-key YOUR_API_KEY            # from cloud.vast.ai/manage-keys
vastai show user                           # confirm auth + funds (see "reading balance" below)

# 1. register SSH key BEFORE creating (applied at container creation)
vastai create ssh-key ~/.ssh/id_ed25519.pub     # or `vastai create ssh-key` to generate one
#   Windows: keys live in C:\Users\<you>\.ssh\ ; generate with: ssh-keygen -t ed25519

# 2. search (section 2) -> note the offer ID from the first column

# 3. create instance (200 GB disk clears the 150 GB floor; PyTorch+CUDA base image)
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 200 \
  --ssh --direct \
  --onstart-cmd "nvidia-smi"
#   -> returns {"success": true, "new_contract": INSTANCE_ID}  (save INSTANCE_ID)
#   storage billing starts at creation; GPU billing starts when status == running

# 4. poll to running
vastai show instance INSTANCE_ID           # status: loading -> running
#   HANDLE exited/unknown/offline -> never reach running; destroy + retry (disk still bills)

# 5. connection details + connect
vastai ssh-url INSTANCE_ID                 # ssh://root@HOST:PORT
ssh root@SSH_HOST -p SSH_PORT              # ssh uses lowercase -p ; scp/sftp use uppercase -P

# 6. move data (either vastai copy, or scp)
vastai copy local:./data/ INSTANCE_ID:/workspace/genomic-variant-classifier/data/
#   or: scp -P PORT file root@HOST:/workspace/...

# 7. teardown (after results transferred back) -- see section 5
vastai destroy instance INSTANCE_ID
```

**Create-instance flags confirmed present in 1.2.0:** `--image`, `--disk` (GB; set at creation, NOT
changeable later), `--ssh`, `--direct` (direct SSH, lower latency than proxy), `--jupyter`,
`--onstart FILE` (script file), `--onstart-cmd "..."` (inline; 16 KB limit — gzip+base64 longer),
`--env` ('quote' env vars + port maps), `--args`.

**Reading balance correctly (this bit me on 2026-07-06):** `vastai show user` shows a `Balance` (cash)
column AND a separate `Credit` ledger. An account funded via CREDIT shows `Balance 0` but
`Billing Creditonly: 1` and `Can Pay: True` — and CAN rent. **`Can Pay: True` is the field that
decides rentability, not `Balance`.** (On 2026-07-06 the account had $25.88 credit and `Can Pay: True`;
the `Balance 0` reading was a false alarm.)

---

## 5. DEPRECATIONS vs OLD SCRIPTS (fix these before relying on them)

| Old form (in 1.0.x scripts) | Current 1.2.0 form | Notes |
|---|---|---|
| `vastai destroy <id>` | `vastai destroy instance <id>` | `instance` subcommand now REQUIRED |
| `echo y \| vastai destroy <id>` | `vastai destroy instance <id>` | **No `-y` flag exists**; destroy is immediate, no prompt — the `echo y \|` pipe is unnecessary. Put your OWN confirmation in the wrapper (see `Vastai_Destroy_Confirmed.ps1`). |
| `vastai show instances` | `vastai show instances` | STILL VALID — not deprecated (now calls the v1 endpoint). `show instances-v1` is a separate parallel subcommand, not a replacement. |
| `cpu_ram>=64000` (assumed MB) | `cpu_ram>=64` (GB) | units bug — section 0 |
| `create instance ... --onstart-cmd` | same | STILL VALID — `--onstart-cmd` and `--onstart FILE` both current |

**When a `-y`/auto-confirm flag DOES exist, Vast lists it explicitly** (e.g. `create ssh-key` has `-y`).
`destroy instance` does NOT list one — so don't script `echo y |` against it; it just destroys.

---

## 6. IRREVERSIBLE COMMANDS (own paste block, after manual verification)

`vastai destroy instance <id>`, `rm -rf`, force-push — never chain these after other commands.
Verify the results are SCP'd back AND the destroy target ID is correct BEFORE running. Use
`Vastai_Destroy_Confirmed.ps1` (prompts + shows the instance before destroying).

## 7. COST DISCIPLINE

- Cap: `dph_total<0.80` in every search; sort `-o 'dph_total'` (cheapest first).
- `stop instance` pauses compute billing but disk keeps billing; `destroy instance` stops ALL billing.
- Run-15 reference: ~11.5 h, ~$6 on a 4090. Budget the full run accordingly; the VM smoke is minutes.
- Record actual billed cost in the postflight doc (POSTFLIGHT_RUN17_PROTOCOL §A/§J).
