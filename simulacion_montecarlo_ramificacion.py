from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResultadoCorrida:
    """Resultado de una trayectoria individual del proceso."""

    estado: str  # "extinguido", "activo_al_limite" o "limite_poblacional"
    generaciones: int
    total_elementos: int
    maximo_generacion: int


def simular_corrida(
    tasa_media: float,
    rng: np.random.Generator,
    generaciones_max: int = 40,
    poblacion_max: int = 100_000,
) -> ResultadoCorrida:
    """Simula una trayectoria usando descendencia Poisson independiente.

    En cada generación, cada elemento produce un número Poisson de
    descendientes con media `tasa_media`. La población de una generación es
    entonces Poisson(tasa_media * población_actual). Los límites solo
    mantienen el ejercicio computacional acotado; alcanzar el límite no se
    interpreta como un resultado físico.
    """
    if tasa_media < 0:
        raise ValueError("tasa_media debe ser no negativa")
    if generaciones_max < 0 or poblacion_max < 1:
        raise ValueError("los límites deben ser válidos")

    poblacion = 1
    total = 1
    maximo = 1

    for generacion in range(1, generaciones_max + 1):
        # El cap evita solicitar números enteros enormes a NumPy.
        media_siguiente = tasa_media * poblacion
        if media_siguiente > poblacion_max:
            return ResultadoCorrida(
                "limite_poblacional", generacion, total, maximo
            )

        siguiente = int(rng.poisson(media_siguiente))
        if siguiente > poblacion_max:
            return ResultadoCorrida(
                "limite_poblacional", generacion, total, maximo
            )

        total += siguiente
        maximo = max(maximo, siguiente)
        poblacion = siguiente
        if poblacion == 0:
            return ResultadoCorrida("extinguido", generacion, total, maximo)

    return ResultadoCorrida(
        "activo_al_limite", generaciones_max, total, maximo
    )


def estimar_extincion_teorica_poisson(tasa_media: float) -> float:
    """Calcula la probabilidad de extinción eventual para Poisson(mu).

    La probabilidad q satisface q = exp(mu * (q - 1)). Se obtiene por
    iteración desde q_0=0, que converge a la menor solución en [0, 1].
    Para mu <= 1, la extinción eventual es 1 (incluido el caso crítico).
    """
    if tasa_media < 0:
        raise ValueError("tasa_media debe ser no negativa")
    if tasa_media <= 1:
        return 1.0

    q = 0.0
    for _ in range(100_000):
        q_nuevo = exp(tasa_media * (q - 1.0))
        if abs(q_nuevo - q) < 1e-13:
            return q_nuevo
        q = q_nuevo
    return q


def intervalo_wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza Wilson aproximado para una proporción (95%)."""
    if n <= 0:
        raise ValueError("n debe ser positivo")
    p = exitos / n
    divisor = 1.0 + z**2 / n
    centro = (p + z**2 / (2.0 * n)) / divisor
    margen = z * sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / divisor
    return max(0.0, centro - margen), min(1.0, centro + margen)


def experimento(
    tasa_media: float,
    n_corridas: int,
    semilla: int,
    generaciones_max: int,
    poblacion_max: int,
) -> list[ResultadoCorrida]:
    rng = np.random.default_rng(semilla)
    return [
        simular_corrida(tasa_media, rng, generaciones_max, poblacion_max)
        for _ in range(n_corridas)
    ]


def resumir(
    tasa_media: float, resultados: Iterable[ResultadoCorrida]
) -> dict[str, float | int | tuple[float, float]]:
    resultados = list(resultados)
    n = len(resultados)
    n_extinguidas = sum(r.estado == "extinguido" for r in resultados)
    n_activas = sum(r.estado == "activo_al_limite" for r in resultados)
    n_limitadas = sum(r.estado == "limite_poblacional" for r in resultados)
    ic = intervalo_wilson(n_extinguidas, n)
    return {
        "tasa_media": tasa_media,
        "n": n,
        "extinguidas": n_extinguidas,
        "activas_al_limite": n_activas,
        "limite_poblacional": n_limitadas,
        "fraccion_extincion_observada": n_extinguidas / n,
        "ic95_extincion": ic,
        "extincion_teorica_eventual": estimar_extincion_teorica_poisson(
            tasa_media
        ),
        "mediana_total": float(np.median([r.total_elementos for r in resultados])),
    }


def imprimir_resultado(resumen: dict[str, float | int | tuple[float, float]]) -> None:
    inferior, superior = resumen["ic95_extincion"]  # type: ignore[misc]
    print(f"\nMedia de descendencia (adimensional): {resumen['tasa_media']:.2f}")
    print(f"Corridas Monte Carlo:                 {resumen['n']:,}")
    print(f"Extinguidas dentro del horizonte:      {resumen['extinguidas']:,}")
    print(f"Activas al llegar a generaciones_max:  {resumen['activas_al_limite']:,}")
    print(f"Detenidas por poblacion_max:           {resumen['limite_poblacional']:,}")
    print(
        "Fracción extinguida observada:         "
        f"{resumen['fraccion_extincion_observada']:.4f} "
        f"(IC 95% Wilson [{inferior:.4f}, {superior:.4f}])"
    )
    print(
        "Extinción eventual teórica del modelo: "
        f"{resumen['extincion_teorica_eventual']:.4f}"
    )
    print(f"Mediana de elementos acumulados:       {resumen['mediana_total']:.1f}")


def graficar_sensibilidad(
    tasas: list[float],
    resumenes_tasa: list[dict[str, float | int | tuple[float, float]]],
    corridas_n: list[int],
    resumenes_n: list[dict[str, float | int | tuple[float, float]]],
    horizontes: list[int],
    extinciones_horizonte: list[float],
) -> None:
    """Muestra sensibilidad a la tasa, el tamaño muestral y el horizonte."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib no está instalado; se omiten los gráficos.")
        return

    tasa_observada = [
        float(r["fraccion_extincion_observada"]) for r in resumenes_tasa
    ]
    tasa_teorica = [
        float(r["extincion_teorica_eventual"]) for r in resumenes_tasa
    ]
    tasa_errores = []
    for r, observado in zip(resumenes_tasa, tasa_observada):
        inferior, superior = r["ic95_extincion"]  # type: ignore[misc]
        tasa_errores.append([observado - inferior, superior - observado])
    tasa_errores = np.asarray(tasa_errores).T

    n_observada = [
        float(r["fraccion_extincion_observada"]) for r in resumenes_n
    ]
    n_errores = []
    anchos_ic = []
    for r, observado in zip(resumenes_n, n_observada):
        inferior, superior = r["ic95_extincion"]  # type: ignore[misc]
        n_errores.append([observado - inferior, superior - observado])
        anchos_ic.append(superior - inferior)
    n_errores = np.asarray(n_errores).T

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].errorbar(
        tasas,
        tasa_observada,
        yerr=tasa_errores,
        fmt="o",
        capsize=4,
        label="Monte Carlo: extinción dentro del horizonte (IC 95%)",
    )
    axes[0].plot(
        tasas,
        tasa_teorica,
        "--",
        label="Extinción eventual teórica",
    )
    axes[0].axvline(1.0, color="gray", linestyle=":", label="Tasa crítica = 1")
    axes[0].set(
        xlabel="Media de descendencia por elemento (adimensional)",
        ylabel="Probabilidad de extinción",
        ylim=(-0.04, 1.04),
        title="Efecto de la tasa media",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        corridas_n,
        n_observada,
        yerr=n_errores,
        fmt="o-",
        capsize=4,
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="Número de corridas Monte Carlo (escala log)",
        ylabel="Extinción observada",
        ylim=(-0.04, 1.04),
        title="Efecto del tamaño muestral",
    )
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.04,
        0.05,
        "Barras: IC 95% Wilson\n"
        + "Anchos: "
        + ", ".join(f"{ancho:.3f}" for ancho in anchos_ic),
        transform=axes[1].transAxes,
        fontsize=8,
    )

    axes[2].plot(horizontes, extinciones_horizonte, "o-", label="Monte Carlo")
    axes[2].axhline(
        1.0,
        color="tab:orange",
        linestyle="--",
        label="Extinción eventual teórica = 1",
    )
    axes[2].set(
        xlabel="Máximo de generaciones consideradas",
        ylabel="Fracción extinguida hasta ese horizonte",
        ylim=(-0.04, 1.04),
        title="Efecto del horizonte (tasa = 1)",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("Sensibilidad del proceso de ramificación abstracto")
    fig.tight_layout()
    plt.show()


def main() -> None:
    # Todos los parámetros son adimensionales y describen solo el proceso
    # matemático; no representan materiales, equipos o magnitudes físicas.
    poblacion_max = 100_000
    generaciones_base = 40
    semilla = 20261122

    print("ANÁLISIS DE SENSIBILIDAD: PROCESO DE RAMIFICACIÓN ABSTRACTO")
    print("Modelo matemático educativo; no representa un arma real.")

    # A. Sensibilidad a la media de descendencia.
    tasas = [round(float(x), 2) for x in np.arange(0.5, 1.51, 0.1)]
    n_tasa = 5_000
    resumenes_tasa = []
    print("\nA) Variación de la tasa media (5 000 corridas por valor)")
    for i, tasa in enumerate(tasas):
        resultados = experimento(
            tasa,
            n_tasa,
            semilla + i,
            generaciones_base,
            poblacion_max,
        )
        resumen = resumir(tasa, resultados)
        resumenes_tasa.append(resumen)
        inferior, superior = resumen["ic95_extincion"]  # type: ignore[misc]
        n_censuradas = resumen["limite_poblacional"]
        print(
            f"  tasa={tasa:.2f}: extinción={resumen['fraccion_extincion_observada']:.3f} "
            f"(IC95% {inferior:.3f}–{superior:.3f}), "
            f"límite poblacional={n_censuradas}"
        )

    # B. Sensibilidad al número de réplicas. El horizonte se mantiene fijo.
    tasa_n = 1.0
    corridas_n = [250, 500, 1_000, 2_500, 5_000, 10_000]
    resumenes_n = []
    print(f"\nB) Variación del número de corridas (tasa={tasa_n:.1f}, horizonte=40)")
    for i, n in enumerate(corridas_n):
        resultados = experimento(
            tasa_n,
            n,
            semilla + 100 + i,
            generaciones_base,
            poblacion_max,
        )
        resumen = resumir(tasa_n, resultados)
        resumenes_n.append(resumen)
        inferior, superior = resumen["ic95_extincion"]  # type: ignore[misc]
        print(
            f"  n={n:>5}: extinción={resumen['fraccion_extincion_observada']:.3f}, "
            f"IC95%=[{inferior:.3f}, {superior:.3f}], "
            f"ancho={superior - inferior:.3f}"
        )

    # C. Sensibilidad al horizonte. Las mismas trayectorias se usan para cada
    # horizonte, lo que facilita comparar la acumulación de extinciones.
    tasa_h = 1.0
    horizontes = [5, 10, 20, 40, 80]
    n_horizonte = 10_000
    max_horizonte = max(horizontes)
    trayectorias = experimento(
        tasa_h,
        n_horizonte,
        semilla + 200,
        max_horizonte,
        poblacion_max,
    )
    extinciones_horizonte = [
        sum(
            r.estado == "extinguido" and r.generaciones <= horizonte
            for r in trayectorias
        )
        / n_horizonte
        for horizonte in horizontes
    ]
    print(f"\nC) Variación del horizonte (tasa={tasa_h:.1f}, {n_horizonte:,} trayectorias)")
    for horizonte, fraccion in zip(horizontes, extinciones_horizonte):
        print(f"  generaciones={horizonte:>2}: fracción extinguida={fraccion:.3f}")

    print(
        "\nInterpretación: la tasa media cambia el comportamiento del proceso; "
        "más corridas reducen la incertidumbre de muestreo; un horizonte mayor "
        "permite observar más extinciones tardías. En la tasa crítica, la "
        "extinción eventual teórica es 1, aunque el experimento finito puede "
        "terminar con trayectorias todavía activas. El límite computacional "
        "solo trunca corridas del modelo abstracto."
    )
    graficar_sensibilidad(
        tasas,
        resumenes_tasa,
        corridas_n,
        resumenes_n,
        horizontes,
        extinciones_horizonte,
    )


if __name__ == "__main__":
    main()
