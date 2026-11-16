terraform {
  required_version = "= 1.9.8"
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "= 1.258.0"
    }
  }
}

# RAM short-lived credentials come only from the provider's environment chain.
provider "alicloud" {
  region = var.deployment.region
}

locals {
  c = var.deployment
  tags = {
    Experiment = local.c.experiment
    ExpiresAt  = local.c.expires_at
    ManagedBy  = "cortex-terraform"
  }
  hourly    = sum([for n in local.c.nodes : n.hourly]) + sum([for d in local.c.disks : d.hourly])
  osd_hosts = [for name, n in local.c.nodes : name if contains(n.roles, "osd")]
}

resource "terraform_data" "gate" {
  input = local.c
  lifecycle {
    precondition {
      condition = (local.hourly > 0 && local.hourly <= local.c.budget_hourly &&
        local.hourly * local.c.ttl_hours + local.c.residual_allowance <= local.c.budget_total &&
        alltrue([for n in local.c.nodes : n.hourly > 0 && n.system_disk_gib >= 40 && n.image_id != "" && n.instance_type != ""]) &&
      alltrue([for d in local.c.disks : d.hourly > 0]))
      error_message = "Approved hourly/TTL total budgets and explicit node/disk quotes required."
    }
    precondition {
      condition = (local.c.pool_min_size >= 1 && local.c.pool_min_size <= local.c.pool_size &&
        local.c.pool_size <= length(local.osd_hosts) &&
        alltrue([for host in local.osd_hosts : contains([for d in local.c.disks : d.node], host)]) &&
        alltrue([for role in ["mon", "mgr", "gateway", "client", "osd"] : contains(flatten([for n in local.c.nodes : n.roles]), role)]) &&
      anytrue([for n in local.c.nodes : contains(n.roles, "client") && !contains(n.roles, "osd") && !contains(n.roles, "gateway")]))
      error_message = "Ceph roles, dedicated client, OSD disks and real replica host domains required."
    }
    precondition {
      condition     = try(timecmp(local.c.expires_at, timeadd(plantimestamp(), "${local.c.ttl_hours}h")) >= 0, false)
      error_message = "Expiry must cover the declared TTL from plan time; tags do not stop billing."
    }
    precondition {
      condition = try(anytrue([for block in ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"] :
        tonumber(split("/", local.c.vpc_cidr)[1]) >= tonumber(split("/", block)[1]) &&
        cidrhost(block, 0) == cidrhost("${cidrhost(local.c.vpc_cidr, 0)}/${split("/", block)[1]}", 0)]) &&
        tonumber(split("/", local.c.subnet_cidr)[1]) >= tonumber(split("/", local.c.vpc_cidr)[1]) &&
      cidrhost(local.c.vpc_cidr, 0) == cidrhost("${cidrhost(local.c.subnet_cidr, 0)}/${split("/", local.c.vpc_cidr)[1]}", 0), false)
      error_message = "Run offline preflight for RFC1918 VPC and contained subnet validation."
    }
  }
}

resource "alicloud_vpc" "experiment" {
  vpc_name   = local.c.experiment
  cidr_block = local.c.vpc_cidr
  tags       = local.tags
  depends_on = [terraform_data.gate]
}

resource "alicloud_vswitch" "private" {
  vpc_id     = alicloud_vpc.experiment.id
  cidr_block = local.c.subnet_cidr
  zone_id    = local.c.zone
  tags       = local.tags
}

resource "alicloud_security_group" "private" {
  name                = local.c.experiment
  vpc_id              = alicloud_vpc.experiment.id
  inner_access_policy = "Drop"
  tags                = local.tags
}

resource "alicloud_security_group_rule" "ssh" {
  type              = "ingress"
  ip_protocol       = "tcp"
  nic_type          = "intranet"
  policy            = "accept"
  port_range        = "22/22"
  security_group_id = alicloud_security_group.private.id
  cidr_ip           = local.c.ssh_cidr
}

# No public address/EIP/NAT. RGW and Ceph are reachable only over the approved VPC route.
resource "alicloud_security_group_rule" "ceph" {
  for_each          = toset(["3300/3300", "6789/6789", "6800/7568", "7480/7480"])
  type              = "ingress"
  ip_protocol       = "tcp"
  nic_type          = "intranet"
  policy            = "accept"
  port_range        = each.value
  security_group_id = alicloud_security_group.private.id
  cidr_ip           = local.c.vpc_cidr
}

resource "alicloud_instance" "node" {
  for_each                   = local.c.nodes
  instance_name              = "${local.c.experiment}-${each.key}"
  availability_zone          = local.c.zone
  instance_type              = each.value.instance_type
  image_id                   = each.value.image_id
  instance_charge_type       = "PostPaid"
  system_disk_category       = "cloud_essd"
  system_disk_size           = each.value.system_disk_gib
  vswitch_id                 = alicloud_vswitch.private.id
  security_groups            = [alicloud_security_group.private.id]
  key_name                   = local.c.key_name
  internet_max_bandwidth_out = 0
  tags                       = merge(local.tags, { Roles = join(",", each.value.roles), CortexABI = "2" })
  depends_on                 = [alicloud_security_group_rule.ssh, alicloud_security_group_rule.ceph]
}

resource "alicloud_ecs_disk" "bluestore" {
  for_each          = local.c.disks
  zone_id           = local.c.zone
  disk_name         = "${local.c.experiment}-${each.key}"
  category          = each.value.category
  performance_level = each.value.performance_level
  size              = each.value.size_gib
  encrypted         = true
  tags              = merge(local.tags, { Purpose = "bluestore-uninitialized" })
  depends_on        = [terraform_data.gate]
}

resource "alicloud_ecs_disk_attachment" "bluestore" {
  for_each             = local.c.disks
  disk_id              = alicloud_ecs_disk.bluestore[each.key].id
  instance_id          = alicloud_instance.node[each.value.node].id
  bootable             = false
  delete_with_instance = false
}

# Metadata only: no user_data, provisioners, install, mkfs, ceph-volume or disk wiping.
output "inventory" {
  value = {
    nodes = { for name, n in alicloud_instance.node : name => {
      id = n.id, private_ip = n.private_ip, roles = local.c.nodes[name].roles
    } }
    disks = { for name, d in alicloud_ecs_disk.bluestore : name => {
      id = d.id, node = local.c.disks[name].node, initialized = false
    } }
    daemon_image    = local.c.daemon_image
    cortex_abi      = local.c.cortex_abi
    ceph_commit     = local.c.ceph_commit
    pool_size       = local.c.pool_size
    pool_min_size   = local.c.pool_min_size
    failure_domain  = local.c.failure_domain
    expected_hourly = local.hourly
    expected_total  = local.hourly * local.c.ttl_hours + local.c.residual_allowance
  }
}
