# app.py - Aplicación en Streamlit: predicción mensual de focos de calor por provincia
# Se ejecuta con:  streamlit run app.py

from pathlib import Path                    # para armar la ruta a los datos sin importar desde dónde se ejecute
import numpy as np                          # para el logaritmo inverso
import pandas as pd                         # para trabajar con tablas
import altair as alt                        # para los gráficos (viene instalado con streamlit)
import streamlit as st                      # la librería de la aplicación web
from sklearn.linear_model import LinearRegression                       # el modelo elegido en la etapa 6
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # métricas

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Focos de calor por provincia", layout="wide")

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
MESES_CORTOS = [m[:3] for m in MESES]       # Ene, Feb, Mar... (para el eje de los gráficos)

# Columnas que NO son características (igual que en el notebook de modelado)
NO_CARACTERISTICAS = ["provincia", "anio", "mes", "focos_calor", "focos_mes_anterior", "log_focos"]

# Nombres de las dos pantallas de la aplicación
MODO_PRONOSTICO = "Pronóstico para enero 2026"
MODO_TECNICO = "Detalle técnico: cómo se probó el modelo"

# Colores de los gráficos
COLOR_BARRAS = "#4c78a8"        # barras normales
COLOR_APAGADO = "#c9d3df"       # barras que no se eligieron (cuando hay una resaltada)
COLOR_RESALTADO = "#e4572e"     # barra de la provincia elegida
COLOR_REAL = "#4c566a"          # línea de lo real
COLOR_MODELO = "#e4572e"        # línea del modelo


def volver_a_focos(valor_log):
    # Pasa de logaritmo a cantidad real de focos (no puede ser negativa)
    return np.maximum(10 ** valor_log - 1, 0)


def grafico_barras(serie, resaltar=None):
    # Ranking de barras de mayor a menor. Si se pasa el nombre de una provincia, esa barra se resalta
    # y las demás quedan en un tono apagado. (st.bar_chart ordena alfabéticamente, por eso se usa altair)
    tabla = serie.reset_index()
    tabla.columns = ["Nombre", "Focos"]
    if resaltar is None:
        color = alt.value(COLOR_BARRAS)
    else:
        color = alt.condition(alt.datum.Nombre == resaltar, alt.value(COLOR_RESALTADO), alt.value(COLOR_APAGADO))
    return alt.Chart(tabla).mark_bar().encode(
        x=alt.X("Nombre", sort="-y", title=None),
        y=alt.Y("Focos", title="Focos de calor"),
        color=color,
        tooltip=["Nombre", alt.Tooltip("Focos", format=",.2f")],      # el cuadro que aparece al pasar el mouse: 2 decimales
    ).properties(width="container", height=350)


def grafico_total_pais(anio, tope):
    # Total del país por mes en un año de prueba: se suman las 24 provincias (real y estimado)
    totales = prueba[prueba["anio"] == anio].groupby("mes")[["focos_calor", "prediccion"]].sum().reset_index()
    tabla = pd.DataFrame({
        "Mes": [MESES_CORTOS[m - 1] for m in totales["mes"]],
        "Real": totales["focos_calor"].values,
        "Estimado por el modelo": totales["prediccion"].values,
    }).melt(id_vars="Mes", var_name="Serie", value_name="Focos")
    series = ["Real", "Estimado por el modelo"]
    return alt.Chart(tabla).mark_line(point=True).encode(
        x=alt.X("Mes", sort=MESES_CORTOS, title=None),
        y=alt.Y("Focos", title="Focos de calor (total del país)", scale=alt.Scale(domain=[0, tope])),   # misma escala en los dos años
        color=alt.Color("Serie", title=None, scale=alt.Scale(domain=series, range=[COLOR_REAL, COLOR_MODELO])),
        strokeDash=alt.StrokeDash("Serie", title=None, scale=alt.Scale(domain=series, range=[[5, 4], [1, 0]])),   # real: línea de puntos
        tooltip=["Mes", "Serie", alt.Tooltip("Focos", format=",.2f")],
    ).properties(width="container", height=300)


def grafico_puntos():
    # Real contra estimado: cada punto es una provincia en un mes de prueba (2024 o 2025)
    tabla = pd.DataFrame({
        "Provincia": prueba["provincia"],
        "Año": prueba["anio"].astype(str),
        "Mes": [MESES[m - 1] for m in prueba["mes"]],
        "Real": prueba["focos_calor"],
        "Estimado": prueba["prediccion"],
    })
    tope = max(tabla["Real"].max(), tabla["Estimado"].max()) * 1.1
    escala = alt.Scale(type="symlog", constant=1, domain=[0, tope])       # escala logarítmica que admite el 0
    marcas = [v for v in [0, 10, 100, 1000, 10000, 100000] if v <= tope]
    puntos = alt.Chart(tabla).mark_circle(size=55, opacity=0.55, color=COLOR_RESALTADO).encode(
        x=alt.X("Real", scale=escala, axis=alt.Axis(values=marcas), title="Focos reales (escala logarítmica)"),
        y=alt.Y("Estimado", scale=escala, axis=alt.Axis(values=marcas), title="Focos estimados por el modelo"),
        tooltip=["Provincia", "Año", "Mes", alt.Tooltip("Real", format=",.0f"), alt.Tooltip("Estimado", format=",.2f")],
    )
    # Línea del acierto perfecto: real = estimado
    diagonal = alt.Chart(pd.DataFrame({"x": [0, tope], "y": [0, tope]})).mark_line(
        color=COLOR_REAL, strokeDash=[6, 4]
    ).encode(x=alt.X("x", scale=escala), y=alt.Y("y", scale=escala))
    return (diagonal + puntos).properties(width="container", height=450)


# ---------------------------------------------------------------------------
# Datos y modelo (se cargan una sola vez gracias al caché de Streamlit)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    carpeta = Path(__file__).parent / "datos" / "procesados"
    entrenamiento = pd.read_csv(carpeta / "train.csv")
    prueba = pd.read_csv(carpeta / "test.csv")
    limpio = pd.read_csv(carpeta / "dataset_limpio.csv")      # dataset completo (llega hasta diciembre de 2025)
    return entrenamiento, prueba, limpio


@st.cache_resource
def entrenar_modelo(entrenamiento):
    # Se entrena la regresión lineal con los datos de entrenamiento (2015-2023)
    caracteristicas = [c for c in entrenamiento.columns if c not in NO_CARACTERISTICAS]
    modelo = LinearRegression()
    modelo.fit(entrenamiento[caracteristicas], entrenamiento["log_focos"])
    return modelo, caracteristicas


entrenamiento, prueba, limpio = cargar_datos()
modelo, CARACTERISTICAS = entrenar_modelo(entrenamiento)

# Predicción para todas las filas de prueba (se usa para gráficos y métricas)
prueba = prueba.copy()
prueba["prediccion"] = volver_a_focos(modelo.predict(prueba[CARACTERISTICAS]))

# Pronóstico de enero de 2026: se arma con los datos de diciembre de 2025 (el último mes con datos)
diciembre = limpio[(limpio["anio"] == 2025) & (limpio["mes"] == 12)].set_index("provincia")
pronostico = pd.DataFrame(0.0, index=diciembre.index, columns=CARACTERISTICAS)       # todas las características en 0
pronostico["log_focos_mes_anterior"] = np.log10(diciembre["focos_calor"] + 1)         # focos del mes anterior (en logaritmo)
for variable in ["T2M", "RH2M", "PRECTOTCORR", "GWETTOP", "GWETROOT"]:
    pronostico[variable + "_mes_anterior"] = diciembre[variable]                      # clima del mes anterior
for provincia_nombre in pronostico.index:
    pronostico.loc[provincia_nombre, "prov_" + provincia_nombre] = 1                  # One-Hot de la provincia
pronostico["mes_1"] = 1                                                               # One-Hot del mes: enero
pronostico = pronostico.reset_index().rename(columns={"index": "provincia"})
pronostico["anio"], pronostico["mes"] = 2026, 1
pronostico["prediccion"] = volver_a_focos(modelo.predict(pronostico[CARACTERISTICAS]))
pronostico["focos_calor"] = np.nan                                                    # todavía no hay valor real
pronostico["focos_mes_anterior"] = diciembre["focos_calor"].values

# Promedio histórico de cada provincia y mes (solo con entrenamiento): sirve como referencia
promedio_historico = entrenamiento.assign(focos=lambda d: d["focos_calor"]).groupby(["provincia", "mes"])["focos"].mean()

# Error medio del modelo con los datos de prueba (2024 y 2025), en focos reales
real = prueba["focos_calor"]
error_modelo = mean_absolute_error(real, prueba["prediccion"])

# ---------------------------------------------------------------------------
# Título y explicación
# ---------------------------------------------------------------------------
st.title("Focos de calor por provincia")
st.write(
    "Esta aplicación estima cuántos **focos de calor** tendrá una provincia en un mes, "
    "usando lo que se sabía hasta el mes anterior (focos y clima). "
    "Sirve para que las brigadas de manejo del fuego **prioricen en qué provincias concentrar recursos**."
)

# ---------------------------------------------------------------------------
# Controles de entrada (barra lateral)
# ---------------------------------------------------------------------------
modo = st.sidebar.radio("¿Qué querés ver?", [MODO_PRONOSTICO, MODO_TECNICO])

if modo == MODO_PRONOSTICO:
    # -----------------------------------------------------------------------
    # Pantalla 1: pronóstico del mes siguiente al último con datos (enero de 2026)
    # Pensada para un usuario que no es técnico
    # -----------------------------------------------------------------------
    provincia = st.sidebar.selectbox("Provincia", ["Todas"] + sorted(prueba["provincia"].unique()), key="prov_pronostico")

    if provincia == "Todas":
        # Con "Todas" se muestra el total del país: la suma de las 24 provincias
        titulo = "Total del país"
        estimado = pronostico["prediccion"].sum()
        mes_anterior = pronostico["focos_mes_anterior"].sum()
        historico = promedio_historico.xs(1, level="mes").sum()
        resaltada = None
    else:
        titulo = provincia
        fila = pronostico[pronostico["provincia"] == provincia].iloc[0]
        estimado = fila["prediccion"]
        mes_anterior = fila["focos_mes_anterior"]
        historico = promedio_historico[(provincia, 1)]
        resaltada = provincia

    st.header(f"{titulo} - Enero de 2026 (pronóstico)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Focos estimados para enero", f"{estimado:,.0f}")
    col2.metric("Focos en diciembre de 2025", f"{mes_anterior:,.0f}")
    col3.metric("Promedio histórico de enero", f"{historico:,.0f}")
    st.caption(
        "Es una estimación hecha con los focos y el clima de diciembre de 2025, el último mes con datos. "
        "Todavía no hay un valor real para comparar."
    )

    # Ranking de provincias: lo que necesita una brigada para priorizar (se resalta la provincia elegida)
    st.subheader("¿Dónde se espera más actividad en enero de 2026?")
    ranking = pronostico.set_index("provincia")["prediccion"].sort_values(ascending=False)
    st.altair_chart(grafico_barras(ranking, resaltar=resaltada))
    if resaltada is None:
        st.caption("Focos estimados por el modelo en cada provincia, de mayor a menor.")
    else:
        st.caption(f"Focos estimados por el modelo en cada provincia, de mayor a menor. En naranja, {resaltada}.")

    # ¿Se puede confiar en el pronóstico? Se probó con 2024 y 2025, años que el modelo no usó para aprender
    st.subheader("¿Podemos confiar en este pronóstico?")
    st.write(
        f"Probamos el modelo con 2024 y 2025, años que no conocía. "
        f"En promedio se equivocó **unos {error_modelo:,.0f} focos** por provincia y mes."
    )
    # Las dos escalas verticales son iguales para poder comparar un año con el otro
    totales = prueba.groupby(["anio", "mes"])[["focos_calor", "prediccion"]].sum()
    tope = totales.to_numpy().max() * 1.08
    for anio_prueba in sorted(prueba["anio"].unique()):
        st.markdown(f"**Total del país en {anio_prueba}**")
        st.altair_chart(grafico_total_pais(anio_prueba, tope))
    st.caption(
        "Cada punto es la suma de las 24 provincias en ese mes. "
        "La línea de puntos es lo que pasó de verdad y la línea continua es lo que había estimado el modelo."
    )

    # Nota final, chica y apagada (no es una caja desplegable)
    st.caption(
        "Estos números son una estimación para priorizar, no una cifra exacta. "
        "Sirven para ver dónde se espera más actividad, no para saber cuántos focos habrá. "
        "Un foco de calor es un punto detectado por satélite y no siempre es un incendio forestal. "
        "Datos: focos de calor (datos.gob.ar) y clima (NASA POWER), 2015-2025."
    )

else:
    # -----------------------------------------------------------------------
    # Pantalla 2: detalle técnico. Validación con 2024 y 2025 (meses que el modelo no usó para aprender)
    # -----------------------------------------------------------------------
    provincia = st.sidebar.selectbox("Provincia", sorted(prueba["provincia"].unique()), key="prov_tecnico")
    st.sidebar.caption("El modelo se entrenó con 2015 a 2023 y se evalúa con 2024 y 2025.")
    anio = st.sidebar.selectbox("Año (datos de prueba)", sorted(prueba["anio"].unique()))
    mes = st.sidebar.selectbox("Mes", range(1, 13), format_func=lambda m: MESES[m - 1])
    fila = prueba[(prueba["provincia"] == provincia) & (prueba["anio"] == anio) & (prueba["mes"] == mes)].iloc[0]

    st.header(f"{provincia} - {MESES[mes - 1]} de {anio}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Focos estimados por el modelo", f"{fila['prediccion']:,.0f}")
    col2.metric("Focos reales registrados", f"{fila['focos_calor']:,.0f}")
    col3.metric("Promedio histórico de ese mes", f"{promedio_historico[(provincia, mes)]:,.0f}")
    st.caption(
        "La estimación se hace con los datos del mes anterior. El valor real se muestra para poder compararlo "
        "(los años 2024 y 2025 no se usaron para entrenar el modelo)."
    )

    # Gráfico 1: evolución de la provincia (real vs estimado)
    st.subheader(f"Evolución de {provincia} en {anio}")
    evolucion = prueba[(prueba["provincia"] == provincia) & (prueba["anio"] == anio)].sort_values("mes")
    grafico = pd.DataFrame({
        "Mes": [MESES_CORTOS[m - 1] for m in evolucion["mes"]],
        "Real": evolucion["focos_calor"].values,
        "Estimado por el modelo": evolucion["prediccion"].values,
    }).melt(id_vars="Mes", var_name="Serie", value_name="Focos")      # una fila por mes y por serie (real / estimado)
    st.altair_chart(
        alt.Chart(grafico).mark_line(point=True).encode(
            x=alt.X("Mes", sort=MESES_CORTOS, title=None),   # orden real de los meses (Ene, Feb, Mar...), no alfabético
            y=alt.Y("Focos", title="Focos de calor"),
            color=alt.Color("Serie", title=None),
            tooltip=["Mes", "Serie", alt.Tooltip("Focos", format=",.2f")],
        ).properties(width="container", height=350)
    )

    # Gráfico 2: ranking de provincias para el mes elegido
    st.subheader(f"¿Dónde se esperaba más actividad en {MESES[mes - 1]} de {anio}?")
    ranking = (prueba[(prueba["anio"] == anio) & (prueba["mes"] == mes)]
               .set_index("provincia")["prediccion"].sort_values(ascending=False))
    st.altair_chart(grafico_barras(ranking))
    st.caption("Focos estimados por el modelo en cada provincia, de mayor a menor.")

    # Métricas del modelo (con todos los meses de prueba) y gráfico de real contra estimado
    st.header("¿Qué tan bien funciona el modelo?")
    error_mes_anterior = mean_absolute_error(real, prueba["focos_mes_anterior"])
    # Referencia "promedio histórico": igual que en el notebook de modelado, el promedio se hace en logaritmo
    # (con solo los datos de entrenamiento) y después se vuelve a focos reales
    promedio_historico_log = entrenamiento.groupby(["provincia", "mes"])["log_focos"].mean()
    historico_en_focos = volver_a_focos(
        pd.Series([promedio_historico_log[(p, m)] for p, m in zip(prueba["provincia"], prueba["mes"])]))
    error_historico = mean_absolute_error(real, historico_en_focos)
    rmse_modelo = np.sqrt(mean_squared_error(real, prueba["prediccion"]))
    r2_log = r2_score(prueba["log_focos"], np.log10(prueba["prediccion"] + 1))

    col_grafico, col_metricas = st.columns([3, 1])
    with col_grafico:
        st.altair_chart(grafico_puntos())
    with col_metricas:
        st.markdown("**Métricas (2024 y 2025)**")
        st.caption(f"**MAE:** {error_modelo:,.0f} focos. Es el error medio por provincia y mes.")
        st.caption(f"**RMSE:** {rmse_modelo:,.0f} focos. Castiga más los errores grandes.")
        st.caption(f"**R²:** {r2_log:.2f} (en logaritmo). Cuanto más cerca de 1, mejor.")
    st.caption(
        "Cada punto es una provincia en un mes de 2024 o 2025. Cuanto más cerca de la línea de puntos "
        "(real = estimado), más acertó el modelo. Las escalas son logarítmicas para ver juntos los meses "
        "con pocos y con muchos focos."
    )
    st.caption(
        f"Para comparar, repetir el mes anterior se equivoca unos {error_mes_anterior:,.0f} focos y el promedio histórico "
        f"de provincia y mes, unos {error_historico:,.0f}. El modelo mejora a las dos, aunque por poco frente al promedio histórico."
    )

    # Nota final, chica y apagada (no es una caja desplegable)
    st.caption(
        "El modelo sirve para estimar qué provincias y meses tendrán más actividad y ordenarlas por prioridad. "
        "No anticipa la cantidad exacta de focos: los errores son mayores en los meses de mucha actividad. "
        "Un foco de calor es un punto detectado por satélite y no siempre es un incendio forestal. "
        "Datos: focos de calor (datos.gob.ar) y clima (NASA POWER), 2015-2025."
    )
