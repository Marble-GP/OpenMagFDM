# Start the local HTML lab. Keep this console open; Ctrl+C stops this server.
$voxelRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$voxelPython = Join-Path $voxelRepo 'build_voxel_deps/python.exe'
$voxelDependencies = Join-Path $voxelRepo 'build_voxel_deps/Library/bin'
if (-not (Test-Path -LiteralPath $voxelPython -PathType Leaf)) {
    throw 'Prepare the local development environment first; see browserlab/README.md.'
}
$voxelPreviousPath = $env:PATH
try {
    $env:PATH = "$voxelDependencies;$env:PATH"
    & $voxelPython (Join-Path $PSScriptRoot 'browserlab/server.py') @args
    $voxelExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $voxelPreviousPath
}
exit $voxelExitCode
