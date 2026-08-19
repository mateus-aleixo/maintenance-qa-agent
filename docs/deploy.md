# Deploying to AWS

The service runs on Lambda behind an HTTP API, from the same container that runs
locally. This is the conformal-rul deployment, repeated: ECR, Lambda, HTTP API
Gateway, Terraform, and GitHub Actions authenticating by OIDC with no stored keys.

Live: `https://245evfkghe.execute-api.eu-west-1.amazonaws.com`

**The generator is deliberately not hosted.** conformal-rul and conformal-seg ship
their own networks. This system's is a 14B model, which does not fit in a Lambda and
is not something a free demo endpoint should pay for per request. What is served is
retrieval and the calibrated gate: `/retrieve`, `/gate`, `/gates`. `/ask` returns 503
with an explanation until an operator sets `LLM_BASE_URL` to any OpenAI-compatible
endpoint, which is a configuration change rather than a different code path.

## What it costs

| Service | Free tier | This project |
|---|---|---|
| Lambda | 1M requests + 400k GB-s / month, forever | ~0 at demo traffic |
| API Gateway (HTTP) | 1M requests / month, first 12 months | ~0; $1/M after year 1 |
| ECR | 500 MB storage, first 12 months | image ≈ 0.46 GB → ~$0.05/month after year 1 |
| CloudWatch logs | 5 GB ingest / month | ~0 |

The stage is throttled to 5 req/s (burst 10) and a $5/month budget alarm emails at
80%, so the worst case is capped twice over.

Measured: **cold start ~8 s, warm well under a second**. Cold start is dominated by
pulling a 456 MB image and loading a 33M-parameter ONNX encoder. The function runs at
2048 MB because Lambda scales vCPU with memory; finishing sooner is not more money.

## Three things that differ from conformal-rul

**The OIDC provider is shared, not created.** An IAM OIDC provider is account-global
and keyed by URL. conformal-rul already created
`token.actions.githubusercontent.com` in this account, so `infra/oidc.tf` here reads
it with a `data` block. Declaring it as a `resource` a second time fails with
`EntityAlreadyExists`, and worse, a `terraform destroy` in this repo would delete the
provider out from under rul's deploy role.

**The registry is not in git.** The ONNX query encoder is 135 MB, so it ships as the
`registry-v1` release asset and `deploy.yml` fetches it before building.

**SQLite had to be taught to run on read-only media.** This one cost three deploys, so
it is worth writing down. The index is baked into the image, and Lambda mounts
everything outside `/tmp` read-only. Three separate things break there:

1. `PRAGMA journal_mode=WAL` creates `-wal` and `-shm` files *next to* the database,
   so merely opening the connection raises `SQLITE_CANTOPEN`.
2. An FTS5 `MATCH` wants scratch space and reaches for the filesystem to get it, so
   even after opening cleanly the *query* fails the same way. This is why `/gates`
   worked while `/retrieve` did not.
3. `journal_mode` is persisted in the database **header**, so a file created in WAL
   mode still demands the sidecars when reopened read-only. `mode=ro` alone is not
   enough; `immutable=1` is what makes SQLite skip the WAL machinery, and it is only
   legal on a database that is not in WAL mode.

So `Store` gained a `read_only` mode that opens `mode=ro&immutable=1` with
`temp_store=MEMORY`, and `registry.py` checkpoints the served copy to
`journal_mode=DELETE` at build time. Both halves are needed. Reproduce the whole class
of failure locally without deploying:

```bash
docker run --rm --read-only --tmpfs /tmp -p 8000:8000 conformal-rag:latest
```

`Store` also keeps **one connection per thread**, because sqlite3 connections cannot
cross threads and FastAPI runs sync endpoints in a worker threadpool.

## Bootstrap from zero

The Lambda references a container image, so ECR must exist and hold one image before
the first full apply:

```powershell
cd infra
terraform init
terraform apply -target=aws_ecr_repository.api -var budget_email=YOU@example.com

$acct = aws sts get-caller-identity --query Account --output text
$reg  = "$acct.dkr.ecr.eu-west-1.amazonaws.com"
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $reg

python -m conformal_seg.registry --run metal_nut   # models/ must exist before build
docker build -t "$reg/conformal-rag:latest" ..
docker push "$reg/conformal-rag:latest"

terraform apply -var budget_email=YOU@example.com
```

`terraform output` prints `api_endpoint`, `ecr_repository_url`,
`lambda_function_name` and `deploy_role_arn`.

## Wire up GitHub deploys

```powershell
gh variable set AWS_ROLE_ARN --body (terraform output -raw deploy_role_arn)
gh variable set AWS_REGION   --body eu-west-1
gh variable set API_ENDPOINT --body (terraform output -raw api_endpoint)
```

After that every deploy is `git tag v0.2.0 && git push --tags`, or
`gh workflow run deploy`.

## Verify

```bash
curl $API/health
curl $API/models
curl -X POST "$API/predict?category=metal_nut" -F image=@part.png
```

The deploy job's smoke test checks `/models` reports `metal_nut`, not just that
`/health` is up: an image built without a registry starts happily and serves 503 on
every prediction, which is a failed deploy that a health check alone calls green.

## Teardown

```powershell
terraform destroy -var budget_email=YOU@example.com
```

The ECR repository is `force_delete`, so images go with it. Note the shared OIDC
provider is a `data` source here and is therefore left alone, which is the intended
behaviour: it belongs to conformal-rul.
