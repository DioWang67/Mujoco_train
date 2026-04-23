param(
    [string]$ProjectSlug = "grasp",
    [string]$Ref = "HEAD",
    [string]$RemoteHost = "root@10.6.243.55",
    [string]$RemoteRoot = "/root/anaconda3/mujoco-train-system",
    [ValidateSet("none", "h1", "grasp")]
    [string]$VerifyProject = "none",
    [switch]$CleanRelease,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing virtualenv Python: $pythonExe"
}

$commit = (& git -C $repoRoot rev-parse --short $Ref).Trim()
if (-not $commit) {
    throw "Failed to resolve git ref: $Ref"
}

$archiveName = "${ProjectSlug}_source_${commit}.tar.gz"
$localArchive = Join-Path $repoRoot "artifacts\sync\$archiveName"
$remoteIncoming = "$RemoteRoot/shared/incoming/$archiveName"
$remoteRelease = "$RemoteRoot/code/releases/$commit"
$remoteCurrent = "$RemoteRoot/code/current"

Write-Host "Repo root      : $repoRoot"
Write-Host "Ref / Commit   : $Ref / $commit"
Write-Host "Project slug   : $ProjectSlug"
Write-Host "Remote host    : $RemoteHost"
Write-Host "Remote root    : $RemoteRoot"
Write-Host "Remote release : $remoteRelease"
Write-Host

$steps = @(
    @{
        Name = "Build archive"
        Action = {
            & $pythonExe -m tools.deploy_release --project-slug $ProjectSlug --ref $Ref
        }
    },
    @{
        Name = "Ensure remote incoming dir"
        Action = {
            & ssh $RemoteHost "mkdir -p $RemoteRoot/shared/incoming"
        }
    },
    @{
        Name = "Upload archive"
        Action = {
            & scp $localArchive "${RemoteHost}:$remoteIncoming"
        }
    },
    @{
        Name = "Remove old release"
        Action = {
            if ($CleanRelease) {
                & ssh $RemoteHost "rm -rf $remoteRelease"
            }
        }
    },
    @{
        Name = "Create release dir"
        Action = {
            & ssh $RemoteHost "mkdir -p $remoteRelease"
        }
    },
    @{
        Name = "Extract release"
        Action = {
            & ssh $RemoteHost "tar xzf $remoteIncoming -C $remoteRelease"
        }
    },
    @{
        Name = "Activate current"
        Action = {
            & ssh $RemoteHost "ln -sfn $remoteRelease $remoteCurrent"
        }
    },
    @{
        Name = "Verify current symlink"
        Action = {
            & ssh $RemoteHost "readlink -f $remoteCurrent"
        }
    }
)

foreach ($step in $steps) {
    if ($DryRun) {
        Write-Host "[dry-run] $($step.Name)"
        continue
    }

    Write-Host "==> $($step.Name)"
    & $step.Action
    Write-Host
}

if ($VerifyProject -eq "h1" -and -not $DryRun) {
    Write-Host "==> Remote smoke verify (h1)"
    & ssh $RemoteHost "cd $RemoteRoot/code/current && MUJOCO_TRAIN_LAYOUT_ROOT=$RemoteRoot MUJOCO_TRAIN_PROJECT_SLUG=h1 /root/anaconda3/bin/python train.py --project h1 --smoke"
}

if ($VerifyProject -eq "grasp" -and -not $DryRun) {
    Write-Host "==> Remote smoke verify (grasp)"
    & ssh $RemoteHost "cd $RemoteRoot/code/current && MUJOCO_TRAIN_LAYOUT_ROOT=$RemoteRoot MUJOCO_TRAIN_PROJECT_SLUG=grasp /root/anaconda3/bin/python train.py --project grasp --smoke --n-envs 1 --fixed-cube"
}
