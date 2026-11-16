# Task 10: offline Ceph+Cortex infrastructure slice

**Actual cloud plan: BLOCKED. Task 10 remains incomplete.** No approved cloud
credentials, region/SKU/image selection, quote or purchase budget is available.
Task 8 has not produced verified Ceph fork binaries. This directory provisions
infrastructure only; it does not establish a working Ceph runtime.

## Files and contract

`main.tf` defines an isolated VPC/vSwitch, private security group, explicitly
selected PostPaid ECS nodes, encrypted separate ESSD BlueStore disks and data-disk
attachments. No data-source API discovery, EIP, NAT, OSS, public database/UI,
provisioner, user-data installer or formatting is included. SSH is limited to
IPv4 `/32`, or an explicitly approved `/28` or narrower range. MON ports
3300/6789, OSD 6800–7568 and RGW 7480 are VPC-only. Admin UI is not opened.
Private routing (existing approved VPN/bastion/private link) must be arranged
separately; an SSH rule does not make private nodes publicly reachable.

`variables.tf` has one **required** `deployment` object with no default. Supply
JSON as `{"deployment": {...}}` in an ignored `approved.tfvars.json`. Run the
offline preflight on exactly the same file before any future Terraform plan.
Never put credentials in it. The object fields and types are defined there;
`test_preflight.fixture()` documents a complete **synthetic test-only** schema,
not an approved configuration, real SKU, image or price. There is deliberately
no ready-to-purchase example. An operator may prepare an unapproved draft with
`approved=false` and zero budgets; preflight will refuse it.

- Select approved `region`, `zone`, exact available `instance_type` and existing
  approved OS `image_id` for every node; no cheapest/latest lookup.
- Choose RFC1918 `vpc_cidr` and contained `subnet_cidr`, existing public-key
  `key_name` (no private key), roles `mon/mgr/gateway/client/osd`, and dedicated
  client host. Gateway and OSD roles are explicit and their costs count. Start
  with one storage host containing MON/MGR/RGW/OSD and one load-client host.
  This is a single-storage-host integration topology, not HA.
- Every OSD host needs at least one distinct **new, uninitialized** data-disk
  resource: `purpose=bluestore`, `initialize=false`, ESSD size/performance level
  and quoted hourly price. Root/system disks never serve BlueStore. Disk IDs,
  not unstable `/dev/vdX` names, are emitted as metadata. No disk wipe, mount,
  ceph-volume, mkfs or automatic initialization is performed.
- Pin the real fork `ceph_commit`, verified `daemon_image` OCI sha256 digest and
  `cortex_abi=2`. These are declarations, **not proof** that the image exists or
  contains verified ABI2 binaries. Use the actual task-8 outputs later, not an
  upstream Ceph image or `rt_osd`. Runtime follow-up must first verify binary
  hashes/ABI, object format, and device identities, then separately obtain disk
  initialization approval. `inventory` and preflight `runtime_config` are
  metadata only, not generated/live `ceph.conf`.
- Explicit `pool_size`, `pool_min_size` and `failure_domain=host` are recorded.
  Replicas cannot exceed OSD host count; size/min_size=1 is only a disposable
  integration test with no host fault tolerance. Multiple OSDs per host are
  not independent physical failure domains.
- Record `approval_ref`, `quote_ref`, currency, positive hourly/total budgets,
  `ttl_hours`, UTC `expires_at`, residual allowance and cleanup plan. Per-node
  `hourly` quotes must include compute, OS image/license, **system disk**, and
  any existing approved access-path costs allocated to this experiment.
  Data disks are separate line items. Total = sum(hourly) × TTL + residual
  allowance. Estimates are user-supplied quotes, not current-price discovery
  or billing enforcement. Build time and storage count; reserve residual
  charges conservatively. Tags do not implement expiry or a spending cap.

## Offline commands

```sh
python3 -m unittest -v test_preflight
python3 preflight.py approved.tfvars.json
terraform fmt -check -diff
```

Preflight makes no network calls and executes no subprocess. A successful
`OFFLINE_VALID` inventory has `cloud_plan=false`; never rename it `plan.json`.
Missing approval, zero/exceeded budget, broad SSH, invalid disk plan, missing
roles, wrong ABI/digest, impossible replicas or expired TTL are rejected.

## Tool pins, download gate and real cloud plan (future authorized work)

- Terraform **1.9.8**, exact pin; official Linux amd64 archive 27,041,434 bytes:
  https://releases.hashicorp.com/terraform/1.9.8/
  SHA256 `186e0145f5e5f2eb97cbd785bc78f21bae4ef15119349f6ad4fa535b83b10df8`,
  checked against the official `terraform_1.9.8_SHA256SUMS` before extraction.
  This is a reproducibility pin, not a claim to be the latest Terraform.
- `aliyun/alicloud` **1.258.0**, exact official released provider pin:
  https://registry.terraform.io/v1/providers/aliyun/alicloud/1.258.0/download/linux/amd64
  Official release metadata reports **59,226,938 bytes** for the Linux amd64
  archive, exceeding this slice's 40 MiB download budget. Not downloaded.
  https://github.com/aliyun/terraform-provider-alicloud/releases/tag/v1.258.0
- Context7 resolve/query used the official `/aliyun/terraform-provider-alicloud`
  docs (initial search returned a DeepWiki index; it was not treated as the
  authoritative release). Version-pinned resource docs are under
  https://github.com/aliyun/terraform-provider-alicloud/tree/v1.258.0/website/docs/r
  Use `ecs_disk`/`ecs_disk_attachment`, not the deprecated disk aliases.

Provider acquisition and schema validation remain blocked by the download gate.
No fabricated `.terraform.lock.hcl` is included. Once provider acquisition is
approved, or an independently verified exact-version local mirror is supplied:

```sh
terraform init -backend=false -input=false
terraform validate
```

Keep the **Terraform-generated** lockfile and review its source/version/checksums.
No successful init/validate is implied by `fmt` parsing. Use least-privilege RAM
short-lived credentials via `ALICLOUD_ACCESS_KEY`, `ALICLOUD_SECRET_KEY` and
`ALICLOUD_SECURITY_TOKEN` environment injection from the operator's secure
session; do not create shell credential files or log environment values. RAM
should be scoped to approved region/resources and required Describe/plan
operations; creating the infrastructure later requires separately approved
create/tag/attach/release permissions. No broad account-admin policy is bundled.

Only after credentials, quote, region/SKU/image, budget and network approval:

```sh
python3 preflight.py approved.tfvars.json
terraform plan -input=false -var-file=approved.tfvars.json -out=tfplan
terraform show -json tfplan > /home/user/OpenCodePrj/.omo/evidence/cortex-plan/task-10/plan.json
```

Those commands are instructions, not evidence of execution. A real plan may
query APIs and reveal quotas/availability failures; do not use `-refresh=false`
or synthetic Terraform tests as a substitute. Protect state, tfplan and the
resulting JSON (resource identifiers and sensitive state may occur). Apply is
outside this slice and requires separate task-13 approval.

## Expiry and state-scoped teardown

1. Before expiry, quiesce clients/OSDs and archive required logs/results to an
   approved private destination; decide snapshot retention and its budget.
2. Use the exact experiment workspace/state and approved variables. Review
   `terraform state list`, then generate a proposed teardown with
   `terraform plan -destroy -input=false -var-file=approved.tfvars.json -out=destroy.tfplan`
   and inspect it with `terraform show destroy.tfplan`. Review exact resource
   IDs, dependencies and data-loss implications. **No destroy is executed here.**
3. An authorized operator may later execute the reviewed saved teardown plan.
   Never enumerate and automatically delete arbitrary resources by tag. Tags
   are correlation hints, not ownership proof or deletion authority.
4. Verify ECS, system disks, separately attached ESSDs, snapshots/auto-snapshots,
   EIPs/public bandwidth, NAT/access routes and archived storage against state,
   actual IDs and billing records. This template creates no EIPs/snapshots, but
   external/manual resources can persist. Data disks explicitly do not vanish
   with instance stop/deletion; state-managed disk release is part of teardown.
   Snapshot retention may continue billing even after disks are released.
5. Reconcile delayed final bills and residual allowance; retain evidence.
   **Stopping ECS is not proof of zero charges**: disks, snapshots, EIPs and
   other retained services may continue to bill. ExpiresAt is only a tag.
