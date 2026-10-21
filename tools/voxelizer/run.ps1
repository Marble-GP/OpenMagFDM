# Local developer launcher. Adds the isolated OCCT environment for this process only.
$voxelRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$voxelExe = Join-Path $voxelRepo 'build_voxelizer/Release/MagVoxelizer.exe'
$voxelDependencies = Join-Path $voxelRepo 'build_voxel_deps/Library/bin'
if (-not (Test-Path -LiteralPath $voxelExe -PathType Leaf)) {
    throw 'Build MagVoxelizer first; see tools/voxelizer/README.md.'
}
$voxelPreviousPath = $env:PATH
try {
    if (Test-Path -LiteralPath $voxelDependencies -PathType Container) {
        $env:PATH = "$voxelDependencies;$env:PATH"
    }
    & $voxelExe @args
    $voxelExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $voxelPreviousPath
}
exit $voxelExitCode
