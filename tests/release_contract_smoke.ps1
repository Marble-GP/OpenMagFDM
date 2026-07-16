param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir
)

$ErrorActionPreference = 'Stop'
$package = (Resolve-Path -LiteralPath $PackageDir).Path
$solver = Join-Path $package 'MagFDMsolver.exe'
if (-not (Test-Path -LiteralPath $solver)) {
    throw "Solver not found: $solver"
}
if (-not (Test-Path -LiteralPath (Join-Path $package 'general_materials.yaml'))) {
    throw 'Bundled general_materials.yaml is missing from the release package'
}

$version = (& $solver --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -ne 'OpenMagFDM 1.6.1') {
    throw "Unexpected --version result: '$version'"
}

Add-Type -AssemblyName System.Drawing
$imagePath = Join-Path $package 'contract_smoke.png'
$bmp = New-Object System.Drawing.Bitmap(200, 200)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::White)
$coil = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Red)
$body = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Gray)
$g.FillRectangle($coil, 92, 92, 16, 16)
$g.FillRectangle($body, 70, 80, 12, 40)
$g.Dispose()
$coil.Dispose()
$body.Dispose()
$bmp.Save($imagePath)
$bmp.Dispose()

function Write-Config([string]$Path, [bool]$Transient, [string]$Start, [string]$End) {
    $lines = @(
        'coordinate_system: cartesian',
        'mesh:',
        '  dx: 1.0e-3',
        '  dy: 1.0e-3',
        'boundary_conditions:',
        '  left:   { type: dirichlet, value: 0.0 }',
        '  right:  { type: dirichlet, value: 0.0 }',
        '  bottom: { type: dirichlet, value: 0.0 }',
        '  top:    { type: dirichlet, value: 0.0 }',
        'materials:',
        '  air:  { rgb: [255, 255, 255], mu_r: 1.0, jz: 0.0 }',
        '  coil: { rgb: [255, 0, 0], mu_r: 1.0, jz: 1.0e6 }',
        '  body: { rgb: [128, 128, 128], mu_r: 2.0, jz: 0.0, calc_force: true }',
        'export:',
        '  precision: double'
    )
    if ($Transient) {
        $lines += @(
            'transient:',
            '  enabled: true',
            '  enable_sliding: true',
            '  total_steps: 0',
            '  slide_direction: vertical',
            "  slide_region_start: $Start",
            "  slide_region_end: $End",
            '  slide_pixels_per_step: 1'
        )
    }
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

Push-Location $package
try {
    $staticConfig = Join-Path $package 'contract_static.yaml'
    Write-Config $staticConfig $false '0' '0'
    & $solver $staticConfig $imagePath 'contract_static_out'
    if ($LASTEXITCODE -ne 0) { throw 'Static contract solve failed' }
    if (-not (Test-Path -LiteralPath 'contract_static_out\Forces\step_0001.csv')) {
        throw 'Static Distributed Amperian Forces CSV was not produced'
    }
    $staticJson = Get-Content -Raw -LiteralPath 'contract_static_out\conditions.json' | ConvertFrom-Json
    if ($staticJson.export.format -ne 'tiff' -or -not $staticJson.export.async) {
        throw 'A partial export block did not inherit TIFF + async defaults'
    }

    # A nonlinear iteration-limit result must remain available for diagnostics
    # but must not be reported to automation as a successful analysis.
    $nonlinearConfig = Join-Path $package 'contract_nonlinear_failure.yaml'
    $nonlinearLines = Get-Content -LiteralPath $staticConfig
    $bodyPattern = '  body: \{ rgb: \[128, 128, 128\], mu_r: 2.0, jz: 0.0, calc_force: true \}'
    $bodyNonlinear = '  body: { rgb: [128, 128, 128], B-H: "mu0 * (1000 / (1 + ($H/1000)^2)) * $H", jz: 0.0, calc_force: true }'
    $nonlinearLines = $nonlinearLines -replace $bodyPattern, $bodyNonlinear
    $nonlinearLines += @(
        'nonlinear_solver:',
        '  enabled: true',
        '  solver_type: newton-krylov',
        '  max_iterations: 2',
        '  tolerance: 0.0',
        '  verbose: true',
        '  anderson:',
        '    enabled: true',
        '    depth: 5'
    )
    Set-Content -LiteralPath $nonlinearConfig -Value $nonlinearLines -Encoding UTF8
    & $solver $nonlinearConfig $imagePath 'contract_nonlinear_failure_out'
    if ($LASTEXITCODE -ne 2) {
        throw "Nonlinear nonconvergence returned $LASTEXITCODE instead of 2"
    }
    $nonlinearLog = Get-Content -Raw -LiteralPath 'contract_nonlinear_failure_out\log.txt'
    if ($nonlinearLog -notmatch 'DID NOT CONVERGE') {
        throw 'Nonlinear nonconvergence was not recorded in log.txt'
    }
    if ($nonlinearLog -notmatch 'Anderson acceleration: enabled \(depth=5, beta=(?:0\.3|3\.0e-01)\)') {
        throw 'Newton-Krylov did not apply the beta=0.3 default when Anderson beta was omitted'
    }
    if ($nonlinearLog -notmatch 'AA (?:accepted|rejected)') {
        throw 'Safeguarded Newton-Krylov Anderson path was not exercised'
    }

    # Invalid mixing parameters must fail at configuration load rather than
    # reaching an unstable nonlinear iteration.
    $invalidAndersonConfig = Join-Path $package 'contract_invalid_anderson.yaml'
    $invalidAndersonLines = $nonlinearLines + '    beta: 1.1'
    Set-Content -LiteralPath $invalidAndersonConfig -Value $invalidAndersonLines -Encoding UTF8
    & $solver $invalidAndersonConfig $imagePath 'contract_invalid_anderson_out'
    if ($LASTEXITCODE -eq 0) {
        throw 'Out-of-range nonlinear_solver.anderson.beta was accepted'
    }

    $metreConfig = Join-Path $package 'contract_metres.yaml'
    Write-Config $metreConfig $true '0.05' '0.09'
    & $solver $metreConfig $imagePath 'contract_metres_out'
    if ($LASTEXITCODE -ne 0) { throw 'Metre metadata run failed' }
    $metreJson = Get-Content -Raw -LiteralPath 'contract_metres_out\conditions.json' | ConvertFrom-Json
    if ($metreJson.openmagfdm_version -ne '1.6.1' -or
        $metreJson.transient.slide_region_units -ne 'm' -or
        [math]::Abs([double]$metreJson.transient.slide_region_start - 0.05) -gt 1e-12) {
        throw 'Decimal slide bounds were not preserved as metres in conditions.json'
    }

    $pixelConfig = Join-Path $package 'contract_pixels.yaml'
    Write-Config $pixelConfig $true '50' '90'
    & $solver $pixelConfig $imagePath 'contract_pixels_out'
    if ($LASTEXITCODE -ne 0) { throw 'Pixel metadata run failed' }
    $pixelJson = Get-Content -Raw -LiteralPath 'contract_pixels_out\conditions.json' | ConvertFrom-Json
    if ($pixelJson.transient.slide_region_units -ne 'pixel' -or
        [int]$pixelJson.transient.slide_region_start -ne 50) {
        throw 'Integer slide bounds did not retain pixel semantics'
    }
} finally {
    Pop-Location
}

Write-Host 'OpenMagFDM v1.6.1 release-contract smoke tests passed.'
