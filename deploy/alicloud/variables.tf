variable "deployment" {
  description = "Explicit approved deployment; same JSON is checked by preflight.py. No purchase defaults."
  type = object({
    approved           = bool
    approval_ref       = string
    region             = string
    zone               = string
    vpc_cidr           = string
    subnet_cidr        = string
    ssh_cidr           = string
    ssh_range_approved = bool
    key_name           = string
    experiment         = string
    expires_at         = string
    ttl_hours          = number
    budget_total       = number
    budget_hourly      = number
    currency           = string
    quote_ref          = string
    residual_allowance = number
    cleanup_plan       = string
    daemon_image       = string
    cortex_abi         = number
    ceph_commit        = string
    pool_size          = number
    pool_min_size      = number
    failure_domain     = string
    nodes = map(object({
      roles           = list(string)
      instance_type   = string
      image_id        = string
      system_disk_gib = number
      hourly          = number
    }))
    disks = map(object({
      node              = string
      purpose           = string
      size_gib          = number
      category          = string
      performance_level = string
      hourly            = number
      initialize        = bool
    }))
  })

  validation {
    condition = (var.deployment.approved && length(trimspace(var.deployment.approval_ref)) > 0 &&
      var.deployment.budget_total > 0 && var.deployment.budget_hourly > 0 &&
      var.deployment.ttl_hours > 0 && var.deployment.residual_allowance >= 0 &&
    length(trimspace(var.deployment.quote_ref)) > 0 && length(trimspace(var.deployment.cleanup_plan)) > 0)
    error_message = "Explicit approval, positive budget/TTL, quote and cleanup plan are required."
  }
  validation {
    condition = try(tonumber(split("/", var.deployment.ssh_cidr)[1]) >= 28 &&
      tonumber(split("/", var.deployment.ssh_cidr)[1]) <= 32 &&
      (endswith(var.deployment.ssh_cidr, "/32") || var.deployment.ssh_range_approved) &&
    can(cidrnetmask(var.deployment.ssh_cidr)), false)
    error_message = "SSH must be IPv4 /32 or approved /28 or narrower."
  }
  validation {
    condition = (var.deployment.cortex_abi == 2 &&
      can(regex("^[^ @]+@sha256:[0-9a-f]{64}$", var.deployment.daemon_image)) &&
    can(regex("^[0-9a-f]{40}$", var.deployment.ceph_commit)) && var.deployment.failure_domain == "host")
    error_message = "Pinned daemon digest, fork commit, ABI2 and host failure domain are mandatory."
  }
  validation {
    condition = alltrue([for d in var.deployment.disks : !d.initialize && d.purpose == "bluestore" &&
      d.size_gib >= 40 && d.category == "cloud_essd" && contains(["PL0", "PL1", "PL2", "PL3"], d.performance_level) &&
    try(contains(var.deployment.nodes[d.node].roles, "osd"), false)])
    error_message = "Only separate uninitialized ESSD BlueStore disks on OSD hosts are supported."
  }
}
