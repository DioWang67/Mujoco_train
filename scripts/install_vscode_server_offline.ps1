param(
    [string]$RemoteHost = "root@10.6.243.55",
    [string]$Commit = "",
    [string]$ArchivePath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

function Get-LocalVSCodeCommit {
    $candidateRoots = @(
        Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code"
    )

    foreach ($root in $candidateRoots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }
        $productJson = Get-ChildItem -LiteralPath $root -Recurse -Filter product.json -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\resources\app\product.json" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($productJson) {
            $product = Get-Content -LiteralPath $productJson.FullName -Raw | ConvertFrom-Json
            if ($product.commit) {
                return [string]$product.commit
            }
        }
    }

    throw "Could not find local VS Code commit. Pass -Commit explicitly."
}

if (-not $Commit) {
    $Commit = Get-LocalVSCodeCommit
}

if (-not $ArchivePath) {
    $ArchivePath = Join-Path $repoRoot "artifacts\vscode-server\vscode-server-$Commit.tar.gz"
}

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    throw "Missing VS Code Server archive: $ArchivePath"
}

$remoteArchive = "/tmp/vscode-server-$Commit.tar.gz"
$remoteInstallDir = "~/.vscode-server/bin/$Commit"

Write-Host "Remote host : $RemoteHost"
Write-Host "Commit      : $Commit"
Write-Host "Archive     : $ArchivePath"
Write-Host

Write-Host "==> Upload VS Code Server archive"
& scp $ArchivePath "${RemoteHost}:$remoteArchive"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host
Write-Host "==> Install VS Code Server on remote"
$installCommand = @"
set -e
mkdir -p $remoteInstallDir
tar -xzf $remoteArchive -C $remoteInstallDir --strip-components=1
chmod -R 755 ~/.vscode-server
test -x $remoteInstallDir/bin/code-server || chmod +x $remoteInstallDir/bin/code-server
echo installed:$remoteInstallDir
"@
& ssh $RemoteHost $installCommand
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host
Write-Host "==> Check shell startup files for visible output"
$checkScript = @'
#!/usr/bin/env bash
set -e

for file in ~/.bashrc ~/.bash_profile ~/.profile; do
  if [ -f "$file" ]; then
    matches="$(grep -nE '^[[:space:]]*echo[[:space:]]|^[[:space:]]*printf[[:space:]]' "$file" || true)"
    if [ -n "$matches" ]; then
      echo "warning: possible visible output in $file"
      printf '%s\n' "$matches"
    fi
  fi
done
echo done
'@
$localCheckScript = Join-Path ([System.IO.Path]::GetTempPath()) "vscode_shell_output_check.sh"
Set-Content -LiteralPath $localCheckScript -Value $checkScript -Encoding ascii
try {
    & scp $localCheckScript "${RemoteHost}:/tmp/vscode_shell_output_check.sh"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & ssh $RemoteHost "bash /tmp/vscode_shell_output_check.sh; rm -f /tmp/vscode_shell_output_check.sh"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Remove-Item -LiteralPath $localCheckScript -ErrorAction SilentlyContinue
}

Write-Host
Write-Host "VS Code Server offline install complete."
Write-Host "Reconnect with VS Code Remote SSH."
