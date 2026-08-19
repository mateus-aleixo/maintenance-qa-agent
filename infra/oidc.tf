# GitHub Actions deploys through OIDC federation, so no long-lived AWS keys
# exist anywhere. The role is scoped to this repository's main branch and
# version tags, and its permissions are the minimum for an app deploy: push the
# image, point the function at it. Infrastructure changes stay a local
# `terraform apply` on purpose.

data "aws_caller_identity" "current" {}

# NOT a resource. An IAM OIDC provider is account-global and keyed by URL, and
# conformal-rul already created this one; declaring it again would fail with
# EntityAlreadyExists, and `terraform destroy` here would tear the provider out
# from under rul's deploy role. The second project onto an account references
# it. If this ever runs against a fresh account, create the provider once by
# applying rul's infra first, or move this block back to a resource.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

locals {
  # "owner/name" + numeric ids -> "owner@oid/name@rid", the identity GitHub
  # actually presents in the token's sub claim.
  github_sub_repo = join("/", [
    "${split("/", var.github_repo)[0]}@${var.github_owner_id}",
    "${split("/", var.github_repo)[1]}@${var.github_repo_id}",
  ])
}

resource "aws_iam_role" "deploy" {
  name = "${var.project}-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = data.aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          # sub format carries pinned numeric ids: repo:owner@id/name@id:ref:...
          "token.actions.githubusercontent.com:sub" = [
            "repo:${local.github_sub_repo}:ref:refs/heads/main",
            "repo:${local.github_sub_repo}:ref:refs/tags/*",
          ]
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "deploy" {
  name = "push-image-and-update-function"
  role = aws_iam_role.deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "EcrPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = aws_ecr_repository.api.arn
      },
      {
        Sid    = "UpdateFunction"
        Effect = "Allow"
        # GetFunctionConfiguration backs `aws lambda wait function-updated`
        Action = [
          "lambda:UpdateFunctionCode",
          "lambda:GetFunction",
          "lambda:GetFunctionConfiguration",
        ]
        Resource = aws_lambda_function.api.arn
      },
    ]
  })
}
