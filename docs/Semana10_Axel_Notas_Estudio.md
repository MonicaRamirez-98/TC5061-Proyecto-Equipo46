# TC5061 — Entregable 7 — Notas personales de estudio (Axel)

**Proyecto:** MiBici, Equipo 46
**Mi parte:** 1) Orquestación de servicios; 2) pipeline CI/CD.
**Integración con Mónica:** pruebas de carga, consumo de recursos, FinOps y consolidación del reporte.

## Qué entendí de esta semana

Algo que me quedó muy claro es que tener una API que funciona en mi computadora no significa que ya tenga un sistema listo para operar. Antes dependíamos de iniciar MLflow por separado, configurar variables de entorno y ejecutar la API manualmente. Si olvidábamos un paso, el modelo podía quedar inaccesible aunque el código estuviera bien. La orquestación busca resolver justamente ese problema.

### 1. Por qué elegí Docker Compose

Para nuestro alcance académico, Compose es suficiente porque la API y MLflow deben comunicarse como un solo sistema. No necesitamos introducir Kubernetes ni Redis/RQ solamente para aparentar una arquitectura más grande. Redis y RQ son opcionales en la actividad, y nuestra API actual responde predicciones síncronas de forma inmediata.

Conservé el código de la API real de la Semana 8 (`api/main.py`) y el servicio que carga el modelo P1 desde MLflow (`mibici/serving/model_loader.py`). No inventé otro endpoint de predicción ni cambié las ocho variables de entrada. La ruta de prueba real sigue siendo `POST /predict`.

La red `backend` hace posible que la API se conecte al servidor por `http://mlflow:5000`. Esto fue importante para mí: dentro de un contenedor, `localhost` significa ese mismo contenedor y **no** la computadora Windows ni otro contenedor.

### 2. Qué pasa cuando se pierde un servicio

Un `healthcheck` permite comprobar si un servicio responde, en lugar de asumir que funciona solo porque inició su proceso. La API devuelve 503 si no pudo cargar el modelo. Por eso decidí que el estado saludable fuera `/health` con respuesta 200, no simplemente tener el puerto 8000 abierto.

La secuencia de arranque es: MLflow saludable → `model-init` importa y verifica el champion → API inicia y queda saludable. Los datos de MLflow persisten en un volumen llamado `mlflow_data`. Apagar la composición con `docker compose down` no elimina el historial; usar `down -v` sí podría borrar ese almacenamiento, por eso no es un comando de limpieza rutinaria.

**Limitación consciente:** `depends_on` administra la secuencia inicial, pero no garantiza una recuperación automática y completa de dependencias ante cualquier fallo posterior. El modelo se carga en memoria al iniciar la API; si cambia el alias, hay que recrear el servicio para adoptar otra versión.

### 3. Cómo resolví la portabilidad del modelo

El Registry de la Semana 8 registró un champion real cuyos artefactos no están en Git. Copiar únicamente un archivo SQLite o cambiar la dirección de MLflow en Docker no resuelve el problema, porque un modelo puede conservar rutas de artefactos del equipo donde fue entrenado.

Por eso añadí un procedimiento reproducible: primero exporto el modelo real por su alias `champion` desde el Registry original, después el servicio `model-init` lo registra en el MLflow del stack. No reentreno el modelo ni altero su firma. Lo valido con las mismas ocho características utilizadas en P1. Esto deja claro qué depende del repositorio y qué depende del artefacto privado del modelo.

### 4. Qué significa CI/CD en nuestro caso

Antes ejecutábamos Pytest en PowerShell y compartíamos capturas para demostrar que las pruebas pasaron. Ahora GitHub Actions vuelve a ejecutar la suite cuando alguien envía cambios. Si una prueba falla, la integración debe detenerse antes de construir o publicar la imagen. Así tenemos una señal automática y verificable del estado de cada commit.

El workflow contiene dos verificaciones obligatorias: la suite completa de Pytest y la construcción del contenedor con prueba integrada de Compose. Esta segunda parte demuestra que los servicios pueden hablar entre ellos y responder a una predicción, no solo que el archivo YAML tiene buena sintaxis. Los reportes JUnit y los logs se guardan como artefactos para facilitar una revisión posterior.

### 5. Qué diferencia hay entre el champion real y el modelo sintético de CI

No sería correcto subir a GitHub los pesos entrenados, ni tampoco llamar «champion real» a un modelo inventado para que las pruebas se vean verdes. En el pipeline genero un **modelo sintético de CI** exclusivamente para verificar arranque, conectividad, registro y contrato de inferencia. **No se usa para medir latencia de nuestro modelo real, para sacar conclusiones de FinOps ni para reportar precisión de MiBici.**

La prueba funcional local sí debe hacerse contra nuestro champion exportado y registrado en Compose. De esto necesitamos capturas auténticas de terminal y GitHub; no debo reportar resultados hasta ejecutarlos.

### 6. Qué aprendí del trabajo en equipo

Como ahora somos dos personas, separé responsabilidades para que nuestros cambios no se pisen. Mi PR incorpora la infraestructura de ejecución y validación. La parte de Mónica comienza sobre esta base: ella podrá dirigir Locust a `http://localhost:8000/predict`, registrar CPU/memoria con `docker stats` y estimar el costo por inferencia con evidencia real. Debemos integrar sus resultados antes de elaborar el PDF final y asegurar que ambos contribuyamos en Git.

## Mini glosario con mis palabras

- **Compose:** especificación que describe cómo se levantan juntos varios servicios.
- **Servicio:** contenedor con una responsabilidad específica, por ejemplo la API o MLflow.
- **Imagen:** plantilla reproducible con el código y las dependencias que necesita el contenedor.
- **Volumen:** almacenamiento que puede persistir aunque detenga los contenedores.
- **Red de Compose:** mecanismo para que los servicios se encuentren por sus nombres.
- **CI:** automatización de validaciones al integrar cambios.
- **CD:** mecanismo para entregar o desplegar una versión después de superar las validaciones.
- **Smoke test:** comprobación breve de las funciones esenciales del sistema.
- **Healthcheck:** prueba de disponibilidad, no equivalente a un benchmark de desempeño.
- **FinOps:** relacionar los recursos consumidos con el valor que obtenemos; lo desarrollará Mónica con mediciones.

## Lo que no debo afirmar sin evidencia

1. No afirmaré que la ejecución de GitHub Actions está verde hasta verla terminada en el repositorio.
2. No afirmaré que Docker Compose funciona con el modelo real hasta que `/health` responda 200 y `/predict` entregue una predicción real.
3. No reportaré consumo de CPU, memoria, latencia ni costo por inferencia sin las mediciones de Locust y `docker stats`.
4. No describiré el job opcional de GHCR como «publicado» si no se activó la variable de publicación.

## Referencias técnicas de consulta

- Docker Inc. *Docker Compose documentation*. https://docs.docker.com/compose/
- Docker Inc. *Control startup and shutdown order in Compose*. https://docs.docker.com/compose/how-tos/startup-order/
- GitHub. *Building and testing Python*. https://docs.github.com/en/actions/tutorials/build-and-test-code/python
- Docker Inc. *Docker Build GitHub Actions*. https://docs.docker.com/build/ci/github-actions/
- MLflow. *Tracking Server*. https://mlflow.org/docs/latest/tracking/server/
- MLflow. *Model Registry Workflows*. https://mlflow.org/docs/latest/ml/model-registry/workflow/
