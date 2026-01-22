
$baseUrl = "http://127.0.0.1:8000"

Write-Host "Testing /classify..."
$classifyBody = @{ text = "This is a legal contract between two parties." } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/classify" -Method Post -Body $classifyBody -ContentType "application/json"
Write-Host "Classify Response: $($response | ConvertTo-Json)"

Write-Host "`nTesting /embed..."
$embedBody = @{ text = "Legal embedding test." } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/embed" -Method Post -Body $embedBody -ContentType "application/json"
Write-Host "Embed Response (truncated): $($response.embedding[0..5] -join ', ')..."

Write-Host "`nTesting /summarize..."
$summaryBody = @{ text = "The quick brown fox jumps over the lazy dog. " * 20 } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/summarize" -Method Post -Body $summaryBody -ContentType "application/json"
Write-Host "Summarize Response: $($response | ConvertTo-Json)"

Write-Host "`nTesting /extract_clauses..."
$clausesBody = @{ text = "1. Confidentiality`nThe parties agree to keep all information confidential." } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "$baseUrl/extract_clauses" -Method Post -Body $clausesBody -ContentType "application/json"
Write-Host "Extract Clauses Response: $($response | ConvertTo-Json)"
