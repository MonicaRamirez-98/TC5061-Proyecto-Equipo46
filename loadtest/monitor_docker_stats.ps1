# Registra el consumo de CPU y memoria de los contenedores del Compose durante una prueba de carga.
# Uso, desde la raíz del repositorio y en una terminal aparte:
#   .\loadtest\monitor_docker_stats.ps1 -Output reports\semana10_locust\docker_stats_u50.csv
# Se detiene con Ctrl+C. Cada fila es una lectura de docker stats para un contenedor.
param(
    [string]$Output = "reports\semana10_locust\docker_stats.csv",
    [int]$IntervalSeconds = 2
)

$folder = Split-Path -Parent $Output
if ($folder -and -not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
"timestamp,container,cpu_percent,mem_usage,mem_percent,net_io,block_io" | Set-Content -Path $Output -Encoding utf8

Write-Host "Registrando docker stats en $Output cada $IntervalSeconds s. Ctrl+C para detener."
while ($true) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" |
        Where-Object { $_ -like "mibici-equipo46-*" } |
        ForEach-Object { "$now,$_" } |
        Add-Content -Path $Output -Encoding utf8
    Start-Sleep -Seconds $IntervalSeconds
}