param(
    [string]$ProjectSlug = "grasp",
    [string]$Ref = "HEAD",
    [string]$RemoteHost = "root@10.6.243.55",
    [string]$RemoteRoot = "/root/anaconda3/mujoco-train-system",
    [string]$VerifyProject = "none",
    [switch]$IncludePrivateAssets,
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
Write-Host "Private assets : $IncludePrivateAssets"
Write-Host

function Invoke-NativeChecked {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-ProjectSmokeArgs {
    param([string]$Slug)

    $configPath = Join-Path $repoRoot "configs\$Slug\project.json"
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Missing project config: $configPath"
    }

    $projectConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if ($null -eq $projectConfig.smoke_args -or $projectConfig.smoke_args.Count -eq 0) {
        return "--smoke"
    }
    return ($projectConfig.smoke_args -join " ")
}

$deployArgs = @("-m", "tools.deploy_release", "--project-slug", $ProjectSlug, "--ref", $Ref)
if ($IncludePrivateAssets) {
    $deployArgs += "--include-private-assets"
}

$steps = @(
    @{
        Name = "Build archive"
        Action = {
            Invoke-NativeChecked -FilePath $pythonExe -ArgumentList $deployArgs
            if (-not (Test-Path -LiteralPath $localArchive)) {
                throw "Archive was not created: $localArchive"
            }
            $archiveInfo = Get-Item -LiteralPath $localArchive
            if ($archiveInfo.Length -le 0) {
                throw "Archive is empty: $localArchive"
            }
        }
    },
    @{
        Name = "Ensure remote incoming dir"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "mkdir -p $RemoteRoot/shared/incoming"
            )
        }
    },
    @{
        Name = "Upload archive"
        Action = {
            Invoke-NativeChecked -FilePath "scp" -ArgumentList @(
                $localArchive,
                "${RemoteHost}:$remoteIncoming"
            )
        }
    },
    @{
        Name = "Verify uploaded archive"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "test -s $remoteIncoming"
            )
        }
    },
    @{
        Name = "Remove old release"
        Action = {
            if ($CleanRelease) {
                Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                    $RemoteHost,
                    "rm -rf $remoteRelease"
                )
            }
        }
    },
    @{
        Name = "Create release dir"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "mkdir -p $remoteRelease"
            )
        }
    },
    @{
        Name = "Extract release"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "tar xzf $remoteIncoming -C $remoteRelease"
            )
        }
    },
    @{
        Name = "Verify extracted release"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "test -f $remoteRelease/train.py && test -f $remoteRelease/sedon_baseline/env.py"
            )
        }
    },
    @{
        Name = "Activate current"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "ln -sfn $remoteRelease $remoteCurrent"
            )
        }
    },
    @{
        Name = "Verify current symlink"
        Action = {
            Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
                $RemoteHost,
                "readlink -f $remoteCurrent"
            )
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

if ($VerifyProject -ne "none" -and -not $DryRun) {
    $smokeArgs = Get-ProjectSmokeArgs -Slug $VerifyProject
    Write-Host "==> Remote smoke verify ($VerifyProject): $smokeArgs"
    Invoke-NativeChecked -FilePath "ssh" -ArgumentList @(
        $RemoteHost,
        "cd $RemoteRoot/code/current && export MUJOCO_TRAIN_LAYOUT_ROOT=$RemoteRoot MUJOCO_TRAIN_PROJECT_SLUG=$VerifyProject MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && /root/anaconda3/bin/python train.py --project $VerifyProject $smokeArgs"
    )
}
