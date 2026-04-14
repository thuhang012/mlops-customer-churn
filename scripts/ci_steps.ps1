param(
    [ValidateSet('install', 'lint', 'test', 'fast', 'dvc-pull', 'data-validate', 'all')]
    [string]$Step = 'fast'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Load-EnvFile {
    param(
        [string[]]$Candidates = @('.env', '.env.example')
    )

    foreach ($candidate in $Candidates) {
        if (-not (Test-Path $candidate)) {
            continue
        }

        Get-Content $candidate | ForEach-Object {
            $line = $_.Trim()
            if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) {
                return
            }

            $parts = $line -split '=', 2
            if ($parts.Length -ne 2) {
                return
            }

            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ([string]::IsNullOrWhiteSpace($name)) {
                return
            }

            if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
                [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }

        return
    }
}

function Run-Install {
    Invoke-Checked 'pip upgrade' { python -m pip install --upgrade pip }
    Invoke-Checked 'pip install requirements' { python -m pip install -r requirements.txt }
}

function Run-Lint {
    Invoke-Checked 'ruff check' { python -m ruff check src tests --select E,F }
}

function Run-Test {
    Invoke-Checked 'pytest' { python -m pytest -m "not ct" tests/ }
}

function Run-DvcPull {
    $username = $env:DAGSHUB_USERNAME
    if ([string]::IsNullOrWhiteSpace($username)) {
        $username = $env:DAGSHUB_USER
    }

    if ([string]::IsNullOrWhiteSpace($username) -or [string]::IsNullOrWhiteSpace($env:DAGSHUB_TOKEN)) {
        throw 'Set DAGSHUB_TOKEN and one of DAGSHUB_USERNAME or DAGSHUB_USER before running dvc-pull.'
    }

    Invoke-Checked 'dvc remote auth basic' { dvc remote modify --local origin auth basic }
    Invoke-Checked 'dvc remote user' { dvc remote modify --local origin user "$username" }
    Invoke-Checked 'dvc remote password' { dvc remote modify --local origin password "$env:DAGSHUB_TOKEN" }
    Invoke-Checked 'dvc pull dataset' { dvc pull data/raw/netflix_large.csv.dvc }
}

function Run-DataValidate {
    if (-not (Test-Path 'data/raw/netflix_large.csv')) {
        throw 'Missing data/raw/netflix_large.csv. Run dvc-pull first.'
    }

    Invoke-Checked 'data validation' { python scripts/validation/validate_data.py data/raw/netflix_large.csv }
}

Load-EnvFile

switch ($Step) {
    'install' {
        Run-Install
    }
    'lint' {
        Run-Lint
    }
    'test' {
        Run-Test
    }
    'fast' {
        Run-Lint
        Run-Test
    }
    'dvc-pull' {
        Run-DvcPull
    }
    'data-validate' {
        Run-DataValidate
    }
    'all' {
        Run-Install
        Run-Lint
        Run-Test
        Run-DvcPull
        Run-DataValidate
    }
}

Write-Host "CI step '$Step' completed." -ForegroundColor Green
