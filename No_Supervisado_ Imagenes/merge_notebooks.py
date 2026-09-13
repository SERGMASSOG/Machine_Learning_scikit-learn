import json
import os
import copy

base = r"c:\Users\masso\Universidad\UNIVERSIDAD LOS ANDES\MACHINE LEARNING DEEPLEARNING IA\Machine Learning\No Supervisado"
source_i = os.path.join(base, "Agrupacion_de_imagenes_I.ipynb")
source_a = os.path.join(base, "Agrupacion_de_imagenes.ipynb")
dest = os.path.join(base, "Agrupacion_de_imagenes_combinado.ipynb")

with open(source_i, "r", encoding="utf-8") as f:
    nb_i = json.load(f)
with open(source_a, "r", encoding="utf-8") as f:
    nb_a = json.load(f)

selected = []
for cell in nb_a["cells"]:
    src = "".join(cell.get("source", []))
    if (
        "def load_images" in src
        or "class ColorClusterPipeline" in src
        or "def comparar_modelos_imagen" in src
        or "pipeline2 = ColorClusterPipeline" in src
        or "pipeline3 = ColorClusterPipeline" in src
        or "pipeline4 = ColorClusterPipeline" in src
        or "KMeans" in src and "silhouette_score" in src
    ):
        selected.append(copy.deepcopy(cell))

final_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Comparación final: HDBSCAN vs KMeans\n",
        "\n",
        "Este notebook conserva en su totalidad el flujo avanzado del notebook del compañero. A la parte final se incorpora el modelo de KMeans con el mismo enfoque de pipeline para comparar ambos métodos bajo la misma metodología.\n",
        "\n",
        "- HDBSCAN se usa para detectar grupos cromáticos densos y descartar ruido visual.\n",
        "- KMeans se usa para comparar una paleta más ordenada y controlada, fijando un número concreto de colores.\n",
        "\n",
        "### Interpretación final\n",
        "\n",
        "Si la imagen requiere una paleta simple, estable y reproducible, KMeans suele ser más útil. Si la imagen presenta muchos matices, sombras o variaciones cromáticas, HDBSCAN suele capturar mejor la distribución real del color.\n",
        "\n",
        "La mejor decisión depende del objetivo del proyecto: si se quiere simplicidad y estructura usar KMeans; si se quiere riqueza cromática y detalle real usar HDBSCAN.\n"
    ]
}

new_nb = copy.deepcopy(nb_i)
new_nb["cells"].extend(selected)
new_nb["cells"].append(final_md)

with open(dest, "w", encoding="utf-8") as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print(f"Notebook creado: {dest}")
print("Celdas base I:", len(nb_i["cells"]))
print("Celdas agregadas del modelo KMeans:", len(selected))
print("Total final:", len(new_nb["cells"]))
