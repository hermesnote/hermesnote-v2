param(
    [ValidateSet('Hardware', 'Progress')]
    [string]$Mode = 'Progress',
    [string]$NasHost = '192.168.0.44',
    [string]$NasUser = 'truenas_admin',
    [string]$ApiBase = 'https://hermesnote.com',
    [string]$JobId = '',
    [switch]$Once
)

# Read-only monitor. No stored passwords, POSTs, service changes, or remote files.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if ($Mode -eq 'Hardware') {
    $Host.UI.RawUI.WindowTitle = 'NAS - CPU / RAM / GPU / VRAM'
    Write-Host 'SSH login and sudo may each ask for your NAS password.' -ForegroundColor Cyan
    Write-Host 'Password input is invisible. Ctrl+C stops the monitor, not training.'
    Write-Host ''
    $dashboard = "date; echo; echo HOST_CPU; LC_ALL=C top -b -n 2 -d 0.2 | grep %Cpu | tail -1; echo; free -h; echo; echo GPU_AND_VRAM; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv; echo; echo TRAINING_CONTAINER_CPU_AND_RAM; docker stats --no-stream hermesnote-training; echo; docker ps --filter name=hermesnote-training --format {{.Names}}:{{.Status}}"
    # sudo applies to watch and every command it runs, for the whole session.
    $remoteCommand = "sudo watch -n 3 '$dashboard'"
    try {
        $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
        & $ssh -t -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$NasUser@$NasHost" $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            Write-Host "SSH monitor exited with code $LASTEXITCODE. This does not mean training failed." -ForegroundColor Yellow
        }
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    Read-Host 'Press Enter to close' | Out-Null
    exit
}

$Host.UI.RawUI.WindowTitle = 'Hermes Training - Epoch / Loss / Accuracy'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ApiBase = $ApiBase.TrimEnd('/')

function Read-Api([string]$Path) {
    try {
        $response = Invoke-RestMethod -Method Get -Uri "$ApiBase/api/model$Path" -TimeoutSec 10
        return $response
    } catch {
        throw "GET $Path : $($_.Exception.Message)"
    }
}

function Format-Metric($Value, [switch]$Percent) {
    if ($null -eq $Value) { return '--' }
    if ($Percent) { return ('{0:F2}%' -f (100 * [double]$Value)) }
    return ('{0:F4}' -f [double]$Value)
}

while ($true) {
    try {
        $selectedId = $JobId
        if (-not $selectedId) {
            $jobs = @(Read-Api '/jobs?limit=20' | Where-Object { $_.job_type -ne 'infer' })
            $active = $jobs | Where-Object { $_.status -in @('pending', 'running') } | Select-Object -First 1
            if ($active) { $selectedId = $active.job_id }
            elseif ($jobs.Count -gt 0) { $selectedId = $jobs[0].job_id }
        }

        if (-not $selectedId) {
            if (-not $Once) { Clear-Host }
            Write-Host 'No training job found. Waiting for a new job...' -ForegroundColor Yellow
        } else {
            $escapedId = [Uri]::EscapeDataString($selectedId)
            $job = Read-Api "/jobs/$escapedId"
            $rows = @(Read-Api "/jobs/$escapedId/progress")
            $models = @($job.graph_spec.nodes | Where-Object { $_.type -eq 'model' })
            if (-not $Once) { Clear-Host }
            Write-Host ('HERMES TRAINING  |  fetched {0:yyyy-MM-dd HH:mm:ss}' -f (Get-Date)) -ForegroundColor Cyan
            Write-Host 'Read-only GET every 5 seconds. Ctrl+C closes monitoring only.'
            if (-not $JobId) { Write-Host 'Auto-follow newest active training; otherwise show latest training record.' }
            Write-Host "Job:    $selectedId"
            Write-Host "Status: $($job.status)    Device: $($job.device)    Phase: $($job.phase)"
            Write-Host "Dates:  $($job.graph_spec.start) ~ $($job.graph_spec.end)"
            Write-Host "Start:  $($job.started_at)    Finish: $($job.finished_at)"
            if ($job.error) { Write-Host "Training error: $($job.error)" -ForegroundColor Red }

            foreach ($model in $models) {
                $nodeRows = @($rows | Where-Object {
                    $rowNode = $_.window_meta.node_id
                    if (-not $rowNode) { $rowNode = $_.node_id }
                    ($rowNode -eq $model.id) -or (-not $rowNode -and $models.Count -eq 1)
                } | Sort-Object epoch)
                $completed = @($nodeRows | Select-Object -ExpandProperty epoch -Unique).Count
                $last = $nodeRows | Select-Object -Last 1
                $target = $model.params.epochs
                $unit = 'epochs'
                if ($model.key -eq 'xgboost') { $target = $model.params.n_estimators; $unit = 'boosting rounds' }
                if ($null -eq $target) { $target = '?' }
                Write-Host ''
                Write-Host "Node $($model.id) [$($model.key)] - Completed $completed / $target $unit" -ForegroundColor Green
                Write-Host "Window=$($model.window)  Batch=$($model.params.batch_size)  Units=$($model.params.units)  Patience=$($model.params.patience)"
                if ($last) {
                    $age = [Math]::Max(0, [int]([DateTimeOffset]::Now - [DateTimeOffset]::Parse($last.created_at)).TotalSeconds)
                    Write-Host "Last report: $($last.created_at) ($age seconds ago)"
                    Write-Host ('Train loss: {0}   Accuracy: {1}' -f (Format-Metric $last.loss), (Format-Metric $last.accuracy -Percent))
                    Write-Host ('Val   loss: {0}   Accuracy: {1}' -f (Format-Metric $last.val_loss), (Format-Metric $last.val_accuracy -Percent))
                    $nodeRows | Select-Object -Last 5 | ForEach-Object {
                        [pscustomobject]@{
                            Epoch = [int]$_.epoch + 1
                            Loss = Format-Metric $_.loss
                            Accuracy = Format-Metric $_.accuracy -Percent
                            ValLoss = Format-Metric $_.val_loss
                            ValAccuracy = Format-Metric $_.val_accuracy -Percent
                        }
                    } | Format-Table -AutoSize | Out-Host
                } else {
                    Write-Host 'Waiting for the first completed epoch. This alone does not mean training is stuck.' -ForegroundColor Yellow
                }
            }
            if ($job.status -in @('done', 'failed')) {
                Write-Host 'This job has ended. The monitor does not submit or restart training.' -ForegroundColor Cyan
            }
        }
    } catch {
        Write-Host ('[{0:HH:mm:ss}] Monitor query failed: {1}' -f (Get-Date), $_.Exception.Message) -ForegroundColor Yellow
        Write-Host 'Training status is unknown; this is not a training failure. Retrying...'
    }
    if ($Once) { break }
    Start-Sleep -Seconds 5
}
