# Interactive viewer; Python/VTK live in the isolated development environment.
$voxelRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$voxelPython = Join-Path $voxelRepo 'build_voxel_deps/python.exe'
$voxelDependencies = Join-Path $voxelRepo 'build_voxel_deps/Library/bin'
if (-not (Test-Path -LiteralPath $voxelPython -PathType Leaf)) {
    throw 'The viewer needs the isolated Python/VTK environment; see tools/voxelizer/README.md.'
}
$voxelPreviousPath = $env:PATH
try {
    $env:PATH = "$voxelDependencies;$env:PATH"
    & $voxelPython (Join-Path $PSScriptRoot 'viewer.py') @args
    $voxelExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $voxelPreviousPath
}
exit $voxelExitCode
