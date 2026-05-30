"""
Generación de gráficas. Todas se guardan en la carpeta output/.
Usa matplotlib (modo Agg, sin interfaz gráfica) y seaborn.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import ruta_output, info

plt.style.use('ggplot')
sns.set_palette("Set2")


def matrices_confusion(y_test, predicciones, prefijo='ml'):
    n = len(predicciones)
    cols = 2 if n >= 4 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11 if cols == 2 else 6, 5 * rows))
    if n == 1:
        axes = [axes]
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes

    for ax, (nombre, y_pred) in zip(axes_flat, predicciones.items()):
        cm = confusion_matrix(y_test, y_pred)
        etiquetas = ['No demora', 'Si demora']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=etiquetas,
                    yticklabels=etiquetas)
        ax.set_title(nombre, fontweight='bold')
        ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')

    for ax in axes_flat[len(predicciones):]:
        ax.set_visible(False)

    plt.suptitle('Matrices de Confusion - Clasificacion', fontweight='bold')
    plt.tight_layout()
    archivo = ruta_output(f'{prefijo}_matrices_confusion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def barras_metricas(resultados_clas, prefijo='ml'):
    fig, ax = plt.subplots(figsize=(11, 6))
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metricas))
    width = 0.2
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    for i, (nombre, vals) in enumerate(resultados_clas.items()):
        valores = [vals[m] for m in metricas]
        bars = ax.bar(x + (i - 1.5) * width, valores, width, label=nombre, color=colors[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metricas)
    ax.set_ylabel('Valor'); ax.set_ylim(0, 1.05)
    ax.set_title('Comparacion de Metricas - Clasificacion', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    archivo = ruta_output(f'{prefijo}_barras_clasificacion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def scatter_regresion(y_test, predicciones, resultados_reg, prefijo='ml'):
    n = len(predicciones)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (nombre, y_pred) in zip(axes, predicciones.items()):
        ax.scatter(y_test, y_pred, alpha=0.5, color='#5B9BD5', edgecolors='white', s=60)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Prediccion perfecta')
        r2 = resultados_reg[nombre].get('R2', resultados_reg[nombre].get('R²', 0))
        ax.set_xlabel('Valor real (ventas_dia)'); ax.set_ylabel('Valor predicho')
        ax.set_title(f"{nombre} (R²={r2:.3f})", fontweight='bold')
        ax.legend()
    plt.tight_layout()
    archivo = ruta_output(f'{prefijo}_scatter_regresion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def barras_errores_regresion(resultados_reg, prefijo='ml'):
    fig, ax = plt.subplots(figsize=(10, 6))
    nombres_r = list(resultados_reg.keys())
    mae_vals = [resultados_reg[n]['MAE'] for n in nombres_r]
    rmse_vals = [resultados_reg[n]['RMSE'] for n in nombres_r]
    x = np.arange(len(nombres_r)); width = 0.35
    bars1 = ax.bar(x - width / 2, mae_vals, width, label='MAE', color='#5B9BD5')
    bars2 = ax.bar(x + width / 2, rmse_vals, width, label='RMSE', color='#ED7D31')
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(nombres_r)
    ax.set_ylabel('Error')
    ax.set_title('Errores MAE y RMSE - Regresion', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    archivo = ruta_output(f'{prefijo}_errores_regresion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def curvas_entrenamiento_dl(hist_basica, hist_profunda, tarea='clasificacion'):
    if tarea == 'clasificacion':
        metrica = 'accuracy'
        metrica_label = 'Accuracy'
    else:
        metrica = None

    n_plots = 2 if metrica else 1
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots))
    if n_plots == 1:
        axes = np.array([axes])

    axes[0, 0].plot(hist_basica['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(hist_basica['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Red Basica - Loss', fontweight='bold'); axes[0, 0].legend()
    axes[0, 1].plot(hist_profunda['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(hist_profunda['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Red Profunda - Loss', fontweight='bold'); axes[0, 1].legend()

    if metrica:
        axes[1, 0].plot(hist_basica[metrica], label='Train', linewidth=2)
        axes[1, 0].plot(hist_basica[f'val_{metrica}'], label='Val', linewidth=2)
        axes[1, 0].set_title(f'Red Basica - {metrica_label}', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 1].plot(hist_profunda[metrica], label='Train', linewidth=2)
        axes[1, 1].plot(hist_profunda[f'val_{metrica}'], label='Val', linewidth=2)
        axes[1, 1].set_title(f'Red Profunda - {metrica_label}', fontweight='bold')
        axes[1, 1].legend()

    plt.suptitle(f'Curvas de Entrenamiento - {tarea.title()}', fontweight='bold')
    plt.tight_layout()
    archivo = ruta_output(f'parte7_curvas_{tarea}.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def comparacion_dl_clasificacion(acc_b, f1_b, tiempo_b, acc_p, f1_p, tiempo_p):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    metricas = ['Accuracy', 'F1-Score']
    vb = [acc_b, f1_b]; vp = [acc_p, f1_p]
    x = np.arange(len(metricas)); w = 0.35
    bars1 = ax[0].bar(x - w/2, vb, w, label='Red Basica', color='#5B9BD5')
    bars2 = ax[0].bar(x + w/2, vp, w, label='Red Profunda', color='#ED7D31')
    for bar in [*bars1, *bars2]:
        ax[0].text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(metricas)
    ax[0].set_ylim(0, 1.05); ax[0].set_title('Metricas - Clasificacion', fontweight='bold'); ax[0].legend()

    tiempos = [tiempo_b, tiempo_p]
    bars = ax[1].bar(['Red Basica', 'Red Profunda'], tiempos, color=['#5B9BD5', '#ED7D31'])
    for bar, t in zip(bars, tiempos):
        ax[1].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05,
                   f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')
    ax[1].set_title('Tiempo de Entrenamiento', fontweight='bold'); ax[1].set_ylabel('Segundos')
    plt.tight_layout()
    archivo = ruta_output('parte7_comparacion_clasificacion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def scatter_regresion_dl(y_test, pred_basica, pred_profunda, r2_b, r2_p):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mn = min(y_test.min(), pred_basica.min(), pred_profunda.min())
    mx = max(y_test.max(), pred_basica.max(), pred_profunda.max())

    axes[0].scatter(y_test, pred_basica, alpha=0.5, color='#5B9BD5', edgecolors='white', s=60)
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Prediccion perfecta')
    axes[0].set_xlabel('Valor real (ventas_dia)'); axes[0].set_ylabel('Valor predicho')
    axes[0].set_title(f'Red Basica (R²={r2_b:.3f})', fontweight='bold'); axes[0].legend()

    axes[1].scatter(y_test, pred_profunda, alpha=0.5, color='#ED7D31', edgecolors='white', s=60)
    axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Prediccion perfecta')
    axes[1].set_xlabel('Valor real (ventas_dia)'); axes[1].set_ylabel('Valor predicho')
    axes[1].set_title(f'Red Profunda (R²={r2_p:.3f})', fontweight='bold'); axes[1].legend()

    plt.tight_layout()
    archivo = ruta_output('parte7_scatter_regresion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")


def comparacion_dl_regresion(r2_b, rmse_b, r2_p, rmse_p):
    fig, ax = plt.subplots(figsize=(10, 5))
    modelos = ['Red Basica', 'Red Profunda']
    r2v = [r2_b, r2_p]; rmsev = [rmse_b, rmse_p]
    x = np.arange(len(modelos)); w = 0.35
    ax2 = ax.twinx()
    b1 = ax.bar(x - w/2, r2v, w, label='R²', color='#5B9BD5')
    b2 = ax2.bar(x + w/2, rmsev, w, label='RMSE', color='#ED7D31')
    for bar, v in zip(b1, r2v):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontweight='bold', color='#5B9BD5')
    for bar, v in zip(b2, rmsev):
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height(),
                 f'{v:,.0f}', ha='center', va='bottom', fontweight='bold', color='#ED7D31')
    ax.set_xticks(x); ax.set_xticklabels(modelos)
    ax.set_ylabel('R²', color='#5B9BD5'); ax2.set_ylabel('RMSE', color='#ED7D31')
    ax.set_title('Comparacion Deep Learning - Regresion', fontweight='bold')
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc='upper left')
    plt.tight_layout()
    archivo = ruta_output('parte7_comparacion_regresion.png')
    plt.savefig(archivo, dpi=150, bbox_inches='tight')
    plt.close()
    info(f"Grafica guardada: {archivo}")
