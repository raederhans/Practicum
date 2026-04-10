[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu",
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Command
)

$ErrorActionPreference = "Stop"

$repoWindows = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoWsl = (& wsl.exe wslpath -a "$repoWindows" 2>$null).Trim()

if (-not $repoWsl) {
    throw "Unable to resolve the repo path inside WSL."
}

function Quote-BashArg {
    param([string]$Value)
    $singleQuoteEscape = "'" + '"' + "'" + '"' + "'"
    return "'" + ($Value -replace "'", $singleQuoteEscape) + "'"
}

$commandBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Command))
$bashCommand = 'set -euo pipefail; cd {0}; source .venv_modeling/bin/activate; tmp=$(mktemp); printf %s {1} | base64 -d > "$tmp"; bash "$tmp"; status=$?; rm -f "$tmp"; exit $status' -f (Quote-BashArg $repoWsl), (Quote-BashArg $commandBase64)

& wsl.exe -d $Distro bash -lc $bashCommand
exit $LASTEXITCODE
