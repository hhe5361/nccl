# API Handoff: NCCL Phase2 Orchestration

Date: 2026-04-26
Repo root: `/mnt/c/Users/hyoeun/dev/nccl`

## Goal

Continue Phase2 experiment automation and reporting work in the API without losing context.

Current target is the `research/code/phase2` workflow.

The user wants:

- 8-worker Phase2 experiments
- master-orchestrated execution from `worker01`
- one fixed rendezvous port for all experiments/modes
- switch logging enabled
- stock vs static-W sweep comparison
- HTML + PNG reporting
- final output equivalence vs STOCK shown in report

## Current Phase2 Design

Phase2 is no longer per-node autonomous looping.

It is now:

- master-only orchestration on `worker01`
- single fixed `MASTER_PORT` per matrix run
- worker single-shot mode execution
- status synchronization through `.ddp_status/<MODE>/<worker>.status`
- `STATUS_DDP` contract inside worker launcher:
  - `1` = running
  - `0` = success right before exit
  - `-1` = failure
- B3 disabled completely in Phase2 runs
- B2 semantic/runtime logic disabled too
- static `PHASE1_STATIC_W` sweep is used instead

## Phase2 Matrix

Main scripts:

- `research/code/phase2/run_b2_matrix.sh`
- `research/code/phase2/run_b2_collective.sh`
- `research/code/phase2/run_b2_mode_worker.sh`
- `research/code/phase2/collective_b2.py`
- `research/code/phase2/compare_b2_vs_stock.py`
- `research/code/phase2/phase2_log_reporter.py`

Current experiment set:

- `allreduce_ring_128mb`
- `allreduce_tree_128mb`
- `allgather_auto_128mb`
- `reducescatter_auto_128mb`
- `alltoall_auto_128mb`

Current modes:

- `stock`
- `b2_w2`
- `b2_w4`
- `b2_w5`
- `b2_w6`
- `b2_w7`
- `b2_w8`

## Important Runtime Behavior

Phase2 runner intentionally disables B2/B3 logic inside NCCL:

- `NCCL_PHASE2_B2_ENABLE=0`
- `NCCL_PHASE3_B3_ENABLE=0`

Meaning:

- `stock` => `PHASE1_STATIC_W=0`
- `b2_wX` => `PHASE1_STATIC_W=X`

So this Phase2 study is a static receiver-window sweep, not semantic B2 and not runtime B3.

## Reporting

Reporter entrypoint:

- `research/code/phase2/phase2_log_reporter.py`

Expected CLI:

```bash
research/venv/bin/python research/code/phase2/phase2_log_reporter.py \
  --input-dir <phase2_matrix_root> \
  --output-dir <phase2_report_root> \
  --switch-log <switch_log_dir>
```

Required report behavior already implemented:

- compare STOCK vs B2 modes
- show throughput
- show latency
- show switch PFC-derived views
- show step/window traces
- save all plots as PNG
- generate HTML report
- show final output equivalence vs STOCK

Final output validation file per experiment:

- `final_output_validation.json`

Worker status files per experiment:

- `.ddp_status/<MODE>/<worker>.status`

## SSH / Topology Model

Worker SSH targets are no longer `worker03@worker03`.

They now resolve through:

- `research/env/network_topology_internal_ips.txt`

Current worker IP mapping:

- `worker01 -> 172.16.0.101`
- `worker02 -> 172.16.0.102`
- `worker03 -> 172.16.0.103`
- `worker04 -> 172.16.0.104`
- `worker05 -> 172.16.0.105`
- `worker06 -> 172.16.0.106`
- `worker07 -> 172.16.0.107`
- `worker08 -> 172.16.0.108`

SSH resolution behavior:

- SSH user:
  - default: worker name itself
  - override: `WORKER_SSH_USER`
  - per-worker override: `WORKER_SSH_USER_MAP`
- SSH host:
  - default: resolved from topology file
  - override file: `NETWORK_TOPOLOGY_FILE`
- SSH port:
  - default: `22`
  - override: `WORKER_SSH_PORT`
  - per-worker override: `WORKER_SSH_PORT_MAP`

## Remote Repo Root Model

This was a real issue and was patched.

Problem:

- master path looked like `/home/worker01/hyoeun/nccl`
- that exact path was incorrectly reused on other workers
- worker02 failed on `cd /home/worker01/hyoeun/nccl`

Current behavior:

- if `REMOTE_REPO_ROOT` is under `/home/<user>/...`
- runner rewrites it to `/home/<resolved_ssh_user>/...` on the remote host

Additional overrides:

- `REMOTE_REPO_ROOT`
- `REMOTE_REPO_ROOT_MAP`

Template tokens supported:

- `{user}`
- `{worker}`

Example:

```bash
export REMOTE_REPO_ROOT='/home/{user}/hyoeun/nccl'
```

## Container Execution Model

Worker-side launch uses:

- `research/code/deploy/run_dev_container.sh`

Important patch:

- do not execute it as `./...`
- call it via `bash ./research/code/deploy/run_dev_container.sh ...`

Reason:

- direct execution hit `Permission denied` in the host environment

## Switch Logging Model

Switch logger is started from the master through the DPU node.

Relevant envs:

- `DPU_NODE_PWD`
- `NETWORK_NODE_PASSWORD`
- `SWITCH_PASSWORD`
- `SWITCH_LOG_ENABLE=1`
- `SWITCH_LOG_INTERVAL_SEC=1`

Port overrides:

- `DPU_NODE_PORT=22`
- `NETWORK_NODE_PORT=7877`

Important detail:

The remote `start_switch_congestion_loggers.sh` did not accept `--network-node-port` as a CLI argument.

This was patched in Phase2 runner by:

- not passing `--network-node-port`
- exporting `NETWORK_NODE_PORT` in the remote DPU shell before running the logger start script

Open question:

- this assumes the remote DPU-side logger script reads `NETWORK_NODE_PORT` from the environment
- if it does not, switch logger startup may still fail later for network-node access

## Latest Observed Failures

These were already diagnosed and partially addressed:

1. Worker SSH connection refused

- Example:
  - `ssh: connect to host 172.16.0.102 port 22: Connection refused`
- Cause:
  - worker SSH port was not necessarily 22
- Patch:
  - added `WORKER_SSH_PORT` and `WORKER_SSH_PORT_MAP`

2. DPU/network-node switch logger port mismatch

- Cause:
  - DPU and network-node ports differ
- Patch:
  - added `DPU_NODE_PORT`
  - pass `NETWORK_NODE_PORT` via DPU remote env

3. `run_dev_container.sh` permission denied

- Cause:
  - host path execution behavior
- Patch:
  - call it through `bash`

4. Wrong repo path on remote workers

- Cause:
  - `/home/worker01/...` reused on worker02+
- Patch:
  - remote repo root now resolves per worker/user

## Most Recent Execution State

The latest user-facing failure before this handoff was:

- switch logger start worked
- worker container prep on `worker01` started
- worker container prep on `worker02` failed due to wrong remote repo root
- that specific issue has since been patched

No fully successful end-to-end 8-worker Phase2 matrix run has been confirmed after the latest patches.

## Recommended Next Step

Run Phase2 again from `worker01` host, not inside the container.

Suggested env setup:

```bash
cd ~/hyoeun/nccl

read -s WORKER_SSH_PASSWORD
export WORKER_SSH_PASSWORD

export WORKER_SSH_PORT=<worker_ssh_port>
export DPU_NODE_PORT=22
export NETWORK_NODE_PORT=7877

export RUN_ID=phase2_b2_matrix_$(date +%y%m%d_%H%M%S)
export MASTER_SERVER=worker01
export MASTER_ADDR=172.16.0.101
export MASTER_PORT=40000
export ALL_WORKERS=worker01,worker02,worker03,worker04,worker05,worker06,worker07,worker08

export DPU_NODE_PWD='<dpu password>'
export NETWORK_NODE_PASSWORD='<network-node password>'
export SWITCH_PASSWORD='<switch password>'
export SWITCH_LOG_ENABLE=1
export SWITCH_LOG_INTERVAL_SEC=1

bash research/code/phase2/run_b2_matrix.sh
```

Optional, if remote repo root auto-rewrite is still wrong:

```bash
export REMOTE_REPO_ROOT='/home/{user}/hyoeun/nccl'
```

## What To Check First If It Fails Again

1. Did switch logger actually start and keep running?
2. Did worker container prep succeed for all 8 workers?
3. Did rank0 open `MASTER_PORT=40000`?
4. Did all worker status files appear under `.ddp_status/...`?
5. Did `final_output_validation.json` get generated per experiment?

## API Prompt Seed

Use the text below as the first message in the API session if needed:

```text
Continue working in /mnt/c/Users/hyoeun/dev/nccl on the Phase2 orchestration and reporting flow.

Focus on research/code/phase2.

Current Phase2 design:
- master-only orchestration on worker01
- single fixed MASTER_PORT
- 8 workers
- static PHASE1 window sweep: stock, b2_w2, b2_w4, b2_w5, b2_w6, b2_w7, b2_w8
- B2/B3 logic in NCCL explicitly disabled for Phase2
- switch logging enabled
- report compares STOCK vs B2 and checks final output equivalence vs STOCK

Important script files:
- research/code/phase2/run_b2_matrix.sh
- research/code/phase2/run_b2_collective.sh
- research/code/phase2/run_b2_mode_worker.sh
- research/code/phase2/collective_b2.py
- research/code/phase2/phase2_log_reporter.py
- research/code/phase2/compare_b2_vs_stock.py

SSH model:
- worker SSH host resolves from research/env/network_topology_internal_ips.txt
- worker SSH user defaults to worker name
- worker SSH port uses WORKER_SSH_PORT / WORKER_SSH_PORT_MAP

Switch logger model:
- DPU_NODE_PORT=22
- NETWORK_NODE_PORT=7877
- NETWORK_NODE_PORT is passed through remote DPU shell env, not CLI

Remote repo root:
- if REMOTE_REPO_ROOT is /home/<user>/..., it auto-rewrites to remote ssh user home
- override available through REMOTE_REPO_ROOT or REMOTE_REPO_ROOT_MAP

Latest known status:
- no confirmed full successful 8-worker Phase2 matrix run after the latest patches
- latest fixes addressed worker SSH host/port, DPU/network-node port handling, noexec container launcher invocation, and remote repo root rewrite

Start by validating current scripts and then continue from the next failing Phase2 matrix run.
```

