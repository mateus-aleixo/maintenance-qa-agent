output "api_endpoint" {
  value       = aws_apigatewayv2_api.http.api_endpoint
  description = "Public base URL of the service"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "lambda_function_name" {
  value = aws_lambda_function.api.function_name
}

output "deploy_role_arn" {
  value       = aws_iam_role.deploy.arn
  description = "Set as AWS_ROLE_ARN repository variable in GitHub"
}
