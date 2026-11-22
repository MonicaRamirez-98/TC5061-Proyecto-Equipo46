# Guía de ejecución — TC5061 Entregable 7 | Equipo 46

**Parte de Axel:** punto 1 (Docker Compose) y punto 2 (CI/CD).
**Base consultada:** rama `main` pública del repositorio del Equipo 46, con FastAPI (`api/main.py`), `Containerfile`, contrato P1 de ocho variables y MLflow Model Registry.
**Estado del paquete:** archivos preparados y verificados sintácticamente; **no afirmar Docker verde ni un workflow ejecutado** hasta que se compruebe en tu Windows y en GitHub.

## 0. Qué necesito tener antes de comenzar

- Docker Desktop instalado y **abierto** (Linux containers).
- PowerShell y Git.
- Repositorio local `TC5061-Proyecto-Equipo46` con acceso a `origin`.
- Entorno `.venv` con Python preferentemente 3.11.
- Registry original de MLflow de la Semana 8 **con el modelo real** `mibici-equipo46-p1-demanda@champion` y artefactos accesibles.
- El ZIP de Axel extraído en una carpeta que identificarás como `$pack`.

> Importante: el artefacto entrenado, `mlflow.db` y `mlruns/` no forman parte de Git. No compartas la BD ni pesos en un commit. La exportación local del modelo es un prerrequisito explícito; el repositorio por sí solo no basta para ejecutar inferencias reales.

## 1. Crear tu rama de trabajo sin afectar el trabajo del equipo

Abre PowerShell en VS Code en la raíz de tu repositorio. Los comandos son:

```powershell
git status --short --branch
git remote -v
git fetch origin
git switch main
git pull origin main
git switch -c feature/semana10-compose-cicd-axel
```

Si `git status` reporta cambios no guardados o Git impide cambiar de rama, **no los elimines**. Guarda primero tu trabajo en una rama/commit o realiza `git stash` solo si entiendes qué quedará guardado. Verifica que `main` incorpora la API real de la Semana 8 antes de continuar:

```powershell
Test-Path .\api\main.py
Test-Path .\mibici\serving\model_loader.py
Test-Path .\Containerfile
Test-Path .\requirements.txt
```

Los cuatro deben devolver `True`. Si no, tu copia local o rama no corresponde a la base revisada; no sobreescribas a ciegas.

## 2. Copiar el paquete a la raíz del proyecto

Cambia únicamente la ruta `$pack` para que apunte a la carpeta del paquete extraído. El ejemplo es ilustrativo:

```powershell
$pack = "C:\RUTA\AL\PAQUETE\TC5061_Semana10_Axel_Parte1y2"
Test-Path "$pack\docker-compose.yml"
```

Debe devolver `True`. Una vez confirmado, copia sin modificar la estructura de carpetas:

```powershell
Copy-Item "$pack\docker-compose.yml" . -Force
Copy-Item "$pack\Containerfile" . -Force
Copy-Item "$pack\requirements-service.txt" . -Force
Copy-Item "$pack\requirements-ci.txt" . -Force
Copy-Item "$pack\.dockerignore" . -Force

New-Item -ItemType Directory -Force .\ops | Out-Null
New-Item -ItemType Directory -Force .\.github\workflows | Out-Null
New-Item -ItemType Directory -Force .\docs | Out-Null
New-Item -ItemType Directory -Force .\tests | Out-Null
Copy-Item "$pack\ops\*" .\ops\ -Force
Copy-Item "$pack\.github\workflows\semana10-ci.yml" .\.github\workflows\ -Force
Copy-Item "$pack\docs\*" .\docs\ -Force
Copy-Item "$pack\tests\*" .\tests\ -Force

python -m ops.update_readme
```

El comando `update_readme` agrega **una sección al final del README existente**; conserva el historial de semanas anteriores. Antes del commit, revisa `git diff -- Containerfile README.md` para comprobar que los cambios son los previstos.

## 3. Dependencias y pruebas locales del código

Activa tu entorno virtual y usa:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt -r requirements-service.txt -r requirements-ci.txt -r requirements-week9.txt
python -m pip install -e .
python -m pytest tests\test_semana10_orquestacion.py -v
python -m pytest tests\ -q
```

La prueba adicional valida YAML, servicios, dependencias, healthchecks y el workflow. La suite completa corresponde al repositorio actual. No modifiques pruebas anteriores solamente para que la terminal muestre `passed`.

**Captura A:** resultado real de las pruebas locales, con comando y resultado visible.

## 4. Recuperar el champion REAL de la Semana 8

Primero verifica el modelo disponible en tu **servidor original** de MLflow. Si todavía está encendido en el puerto 5000, conserva esa terminal. Si no está encendido, inicia el MLflow original de la Semana 8 apuntando a **su base y a sus artefactos reales**, no a una SQLite vacía:

```powershell
# Ejemplo SOLO si la base mlflow.db ORIGINAL está en la raíz actual y sus artefactos son accesibles.
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

En otra terminal con `.venv` activado y situada en el proyecto:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m mibici.serving.verify_model
python -m ops.export_champion
Test-Path .\models\compose-champion\MLmodel
```

El último comando debe devolver `True`. Si falla la verificación, no uses el modelo de demostración como sustituto. Recupera el Registry y los artefactos verdaderos antes de continuar. Si el directorio ya existe y quieres **reexportar** conscientemente, ejecuta `python -m ops.export_champion --overwrite`.

**A continuación detén el MLflow ORIGINAL** con `Ctrl+C` en su terminal para liberar el puerto 5000. Compose iniciará otro MLflow aislado que recibe el modelo exportado. La base original no se sobrescribe.

**Captura B:** `MLmodel` encontrado + nombre del alias verificado (sin subir artefactos).

## 5. Levantar el sistema completo con Docker Compose

Comprueba primero el estado del motor Docker y la configuración:

```powershell
docker version
docker compose version
docker compose -f docker-compose.yml config --quiet
```

Después ejecuta **un único comando de arranque**:

```powershell
docker compose -f docker-compose.yml up -d --build
```

El arranque sigue la dependencia: MLflow saludable → `model-init` completa el registro real → FastAPI. Consulta:

```powershell
docker compose -f docker-compose.yml ps --all
docker compose -f docker-compose.yml logs --tail 100 mlflow model-init api
```

Esperamos observar `mlflow` en estado `healthy`, `model-init` terminado correctamente y `api` en estado `healthy`. Si `model-init` falla, primero inspecciona su log; no presentes como exitosa una API que responde 503.

**Captura C:** `docker compose ps --all` con los tres servicios y estado.

## 6. Comprobar la inferencia y la versión de modelo

En la misma terminal, sin depender de herramientas de terceros:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
python -m ops.smoke_test --url http://localhost:8000
```

El último programa prueba una predicción con las **ocho variables reales** de P1, comprueba que el modelo cargó y compara la versión reportada con `/model-info`. Comprueba además:

- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

**Captura D:** resultado `SMOKE TEST APROBADO` sobre champion REAL (no el demo del CI).

> La API carga el modelo solo al iniciar. Si vuelves a exportar otro champion, ejecuta `docker compose run --rm model-init` y después `docker compose up -d --force-recreate api` para tomar el nuevo alias. Evita borrar el volumen persistente si tiene información que quieras conservar.

## 7. Validar el pipeline GitHub Actions

Antes de subir:

```powershell
git status --short
git diff --stat
git diff -- Containerfile README.md
git check-ignore .\models\compose-champion\MLmodel
```

El último comando debe mostrar que el artefacto está ignorado. Incluye **solo el trabajo de tu parte**:

```powershell
git add docker-compose.yml Containerfile .dockerignore requirements-service.txt requirements-ci.txt ops/ .github/workflows/semana10-ci.yml docs/Semana10_Axel_Guia_Paso_a_Paso.md docs/Semana10_Axel_Notas_Estudio.md tests/test_semana10_orquestacion.py README.md
git commit -m "feat(semana10): orquesta MLflow API y automatiza CI con Compose"
git push -u origin feature/semana10-compose-cicd-axel
```

Abre GitHub en el repositorio central `iplatamx/TC5061-Proyecto-Equipo46`, sección **Pull requests > New pull request**, con rama base `main` y rama de comparación `feature/semana10-compose-cicd-axel` (o el nombre equivalente en tu fork). Usa el título: **Semana 10 | Axel | Docker Compose + CI/CD MiBici**.

En **Actions** verifica dos jobs obligatorios:

1. `1 | Pruebas del proyecto` (Pytest sobre `tests/`).
2. `2 | Imagen OCI + Compose + inferencia smoke` (construcción, healthchecks e inferencia con modelo sintético SOLO para CI).

El tercer job de publicación GHCR permanece `skipped` salvo que se habilite `PUBLISH_GHCR=true` a nivel de repositorio. No es un error: la publicación al registro es una mejora opcional de la rúbrica, no requisito indispensable del punto 2. Si el equipo desea activarla, primero revisen permisos de Actions y paquetes del repositorio; no configuren credenciales externas en el YAML.

**Captura E:** ejecución verde real en GitHub con nombre de workflow, rama y commit.
**Captura F:** Pull Request mostrando archivos modificados y autoría.

**Importante:** los tests locales y la validación sintáctica NO equivalen a una ejecución verde en GitHub. Si un job falla, abre el paso rojo y corrige el error antes de pedir el merge.

## 8. Entregar la base a Mónica

Cuando tu PR esté revisado y, de preferencia, integrado en `main`, Mónica debe actualizar su rama desde `main`, preparar **su propio acceso al champion real** y levantar Compose con las instrucciones del README. Después podrá probar el endpoint `POST http://localhost:8000/predict` y observar recursos con:

```powershell
docker stats --no-stream
```

Mónica se encargará de los puntos **3, 4 y 5** (Locust, consumo de recursos, costo por inferencia). Entre ambos integrarán el **punto 6**: README, repositorio, evidencia de Git y reporte PDF final `TC5061_Operacion_FinOps_Equipo46.pdf`.

Tu evidencia debe separar los resultados del modelo real (locales) de los checks con modelo sintético de GitHub Actions. Este último demuestra integración, **no** mide la calidad ni los costos reales del modelo MiBici.

## 9. Problemas frecuentes y su solución

| Situación | Diagnóstico / acción |
|---|---|
| `docker` no se reconoce | Verifica instalación y ejecución de Docker Desktop. |
| Puerto 5000 ocupado | Detén MLflow original antes de `docker compose up`. |
| Puerto 8000 ocupado | Detén API local anterior o configura `API_PORT=8001` en `.env` local (no versionado). |
| `model-init` dice que falta `MLmodel` | Exporta el champion real y confirma `models/compose-champion/MLmodel`. |
| API devuelve 503 | Consulta logs de `model-init` y `api`; confirma alias y acceso a artefactos en MLflow. |
| YAML inválido | `docker compose config --quiet` indica el error de configuración. |
| README se duplicó | `ops.update_readme` está protegido contra duplicación por encabezado. |
| CI falla en Pytest | Corrige la prueba o dependencia desde el registro de Actions; no silencies fallas. |
| Se ejecutó `down -v` | El volumen y sus modelos pueden perderse; reimporta desde el champion real. |
| Mónica obtiene predicciones diferentes | Verifiquen la versión reportada por `/model-info`; una instancia puede haber cargado otra versión al inicio. |

## 10. Lista de comprobación del 100% para tu parte

- [ ] Hay un `docker-compose.yml` con API, MLflow y etapa de registro del modelo.
- [ ] Se usa un solo comando para iniciar el stack una vez aportado el artefacto real.
- [ ] Redes, healthchecks, `depends_on`, volumen persistente y entorno están configurados.
- [ ] API `/health` en `200` y `/predict` con champion real, demostrado localmente.
- [ ] Workflow activado por `push`, `pull_request` y ejecución manual.
- [ ] Pytest se ejecuta automáticamente y conserva reporte JUnit.
- [ ] La imagen se construye y Compose pasa una prueba integrada automatizada.
- [ ] Workflow en GitHub terminado en verde (captura auténtica).
- [ ] README existente actualizado y el PR registra tu contribución.
- [ ] Mónica recibe endpoint, contrato de ocho variables y guía para sus mediciones.

**Alcance y transparencia:** un ZIP no puede demostrar por sí solo que Docker está en ejecución ni que GitHub Actions salió verde. Esos dos resultados exigen ejecutar los comandos en tu entorno y revisar el workflow publicado.
