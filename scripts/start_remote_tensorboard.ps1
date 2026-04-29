param(
    [string]$RemoteHost = "root@10.6.243.55",
    [string]$RemoteRoot = "/root/anaconda3/mujoco-train-system",
    [string]$ProjectSlug = "",
    [string]$JobName = "",
    [string]$Port = "",
    [switch]$LatestRun
)

$ErrorActionPreference = "Stop"

function Invoke-SshCapture {
    param(
        [string]$Command
    )
    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        $process = Start-Process `
            -FilePath "ssh" `
            -ArgumentList @($RemoteHost, $Command) `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        $output = @()
        if (Test-Path $stdoutFile) {
            $output += Get-Content $stdoutFile
        }
        if (Test-Path $stderrFile) {
            $output += Get-Content $stderrFile
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output = @($output)
        }
    }
    finally {
        Remove-Item $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    }
}

function Invoke-SshBackgroundStart {
    param(
        [string]$Command
    )
    $process = Start-Process `
        -FilePath "ssh" `
        -ArgumentList @("-f", $RemoteHost, $Command) `
        -NoNewWindow `
        -Wait `
        -PassThru
    return $process.ExitCode
}

function Test-LocalPortAvailable {
    param(
        [int]$CandidatePort
    )
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Parse("127.0.0.1"),
            $CandidatePort
        )
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Test-RemotePortAvailable {
    param(
        [int]$CandidatePort
    )
    $probeResult = Invoke-SshCapture -Command "bash -lc 'echo > /dev/tcp/127.0.0.1/$CandidatePort' >/dev/null 2>&1"
    return ($probeResult.ExitCode -ne 0)
}

function Get-TensorBoardPort {
    param(
        [string]$SelectedProject,
        [string]$RequestedPort
    )

    if ($RequestedPort) {
        $parsedPort = 0
        if (-not [int]::TryParse($RequestedPort, [ref]$parsedPort) -or $parsedPort -le 0) {
            throw "Invalid port: $RequestedPort"
        }
        return $parsedPort
    }

    $preferredPorts = @()
    if ($SelectedProject -ieq "h1") {
        $preferredPorts += 6006
    }
    elseif ($SelectedProject -ieq "grasp") {
        $preferredPorts += 6007
    }
    $preferredPorts += 6006..6025

    foreach ($candidatePort in ($preferredPorts | Select-Object -Unique)) {
        if ((Test-LocalPortAvailable -CandidatePort $candidatePort) -and
            (Test-RemotePortAvailable -CandidatePort $candidatePort)) {
            return $candidatePort
        }
    }
    throw "No available TensorBoard port found in 6006-6025."
}

function Get-Selection {
    param(
        [string]$Title,
        [string[]]$Options
    )

    if (-not $Options -or $Options.Count -eq 0) {
        throw "${Title}: no options available."
    }
    if ($Options.Count -eq 1) {
        return $Options[0]
    }

    Write-Host $Title
    for ($i = 0; $i -lt $Options.Count; $i++) {
        Write-Host ("[{0}] {1}" -f ($i + 1), $Options[$i])
    }
    while ($true) {
        $raw = Read-Host "Select 1-$($Options.Count)"
        $selectedIndex = 0
        if ([int]::TryParse($raw, [ref]$selectedIndex) -and $selectedIndex -ge 1 -and $selectedIndex -le $Options.Count) {
            return $Options[$selectedIndex - 1]
        }
        Write-Host "Invalid selection."
    }
}

function Get-RemoteProjects {
    $runsRoot = "$RemoteRoot/runs"
    $command = "find '$runsRoot' -mindepth 1 -maxdepth 1 -type d -printf '%f`n' 2>/dev/null | sort"
    $result = Invoke-SshCapture -Command $command
    if ($result.ExitCode -ne 0) {
        throw "Failed to list remote projects."
    }
    return @(
        $result.Output |
        Where-Object { $_ -and $_.ToString().Trim() } |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ -match '^[A-Za-z0-9][A-Za-z0-9_-]*$' -and -not $_.StartsWith("--") }
    )
}

function Get-RemoteJobs {
    param(
        [string]$SelectedProject
    )
    $tbRoot = "$RemoteRoot/runs/$SelectedProject/logs/tb"
    $command = "find '$tbRoot' -mindepth 1 -maxdepth 1 -type d -printf '%f`n' 2>/dev/null | sort"
    $result = Invoke-SshCapture -Command $command
    if ($result.ExitCode -ne 0) {
        throw "Failed to list remote jobs for project '$SelectedProject'."
    }
    return @(
        $result.Output |
        Where-Object { $_ -and $_.ToString().Trim() } |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ -match '^[A-Za-z0-9][A-Za-z0-9_-]*$' -and -not $_.StartsWith("--") }
    )
}

function Get-LatestTensorBoardRun {
    param(
        [string]$SelectedProject,
        [string]$SelectedJob
    )
    $tbRoot = "$RemoteRoot/runs/$SelectedProject/logs/tb/$SelectedJob"
    $command = "find '$tbRoot' -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f`n' 2>/dev/null | sort -n | tail -n 1 | awk '{print `$2}'"
    $result = Invoke-SshCapture -Command $command
    if ($result.ExitCode -ne 0) {
        throw "Failed to find latest TensorBoard run under project '$SelectedProject' job '$SelectedJob'."
    }
    $latestRun = @(
        $result.Output |
        Where-Object { $_ -and $_.ToString().Trim() } |
        ForEach-Object { $_.ToString().Trim() } |
        Where-Object { $_ -match '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -and -not $_.StartsWith("--") }
    ) | Select-Object -Last 1

    if (-not $latestRun) {
        throw "No TensorBoard run directories found under: $tbRoot"
    }
    return "$SelectedJob/$latestRun"
}

if (-not $ProjectSlug) {
    $ProjectSlug = Get-Selection -Title "Available projects:" -Options (Get-RemoteProjects)
}

if (-not $JobName) {
    $jobs = Get-RemoteJobs -SelectedProject $ProjectSlug
    if ($jobs.Count -eq 0) {
        $JobName = $ProjectSlug
    } else {
        $JobName = Get-Selection -Title "Available jobs for project '$ProjectSlug':" -Options $jobs
    }
}

if ($LatestRun) {
    $JobName = Get-LatestTensorBoardRun -SelectedProject $ProjectSlug -SelectedJob $JobName
    Write-Host "Selected latest TensorBoard run: $JobName"
}

$portNumber = Get-TensorBoardPort -SelectedProject $ProjectSlug -RequestedPort $Port

$remoteLogDir = "$RemoteRoot/runs/$ProjectSlug/logs/tb/$JobName"
$safeJobName = ($JobName -replace '[\\/]', '_' -replace '[^A-Za-z0-9_.-]', '_')
$remoteLog = "/tmp/tb_${ProjectSlug}_${safeJobName}_${portNumber}.log"

Write-Host "Starting TensorBoard on remote for project=$ProjectSlug job=$JobName..."
$remoteLauncher = "$RemoteRoot/code/current/scripts/start_remote_tensorboard.sh"
$startResult = Invoke-SshCapture -Command "bash '$remoteLauncher' '$RemoteRoot' '$ProjectSlug' '$JobName' '$portNumber'"
foreach ($line in $startResult.Output) {
    if ($line) {
        Write-Host $line
    }
}
if ($startResult.ExitCode -ne 0) {
    Write-Host "Remote TensorBoard failed to start."
    exit $startResult.ExitCode
}

Write-Host "TensorBoard ready at 127.0.0.1:$portNumber"
Write-Host "Opening tunnel: localhost:$portNumber -> remote:$portNumber"
Write-Host "Then open: http://localhost:$portNumber"
Write-Host "Remote log: $remoteLog"
Write-Host

& ssh -N -L "${portNumber}:127.0.0.1:${portNumber}" $RemoteHost
exit $LASTEXITCODE
