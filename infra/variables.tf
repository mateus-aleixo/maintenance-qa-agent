variable "project" {
  description = "Name prefix for every resource"
  type        = string
  default     = "conformal-rag"
}

variable "region" {
  description = "AWS region (eu-west-1 keeps latency low from Portugal, and is where conformal-rul lives)"
  type        = string
  default     = "eu-west-1"
}

variable "github_repo" {
  description = "GitHub repository allowed to deploy via OIDC, as owner/name"
  type        = string
  default     = "mateus-aleixo/conformal-rag"
}

# GitHub's OIDC sub claim pins numeric account/repo ids (owner@id/repo@id)
# so a deleted-and-recreated repo of the same name cannot assume the role.
# Find them: gh api users/<owner> --jq .id / gh api repos/<owner>/<name> --jq .id
variable "github_owner_id" {
  description = "Numeric GitHub account id baked into the OIDC sub claim"
  type        = string
  default     = "75174997"
}

variable "github_repo_id" {
  description = "Numeric GitHub repository id baked into the OIDC sub claim"
  type        = string
  default     = "1317839103"
}

variable "image_tag" {
  description = "Image tag the Lambda points at; CI moves the function to new tags"
  type        = string
  default     = "latest"
}

variable "budget_email" {
  description = "Email for the monthly cost-budget alarm"
  type        = string
}
