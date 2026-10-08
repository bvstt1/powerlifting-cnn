# BP_FRONT PoseCNN: Fault Code and Pose Observability

## 1. Objetivo y alcance

Este es un analisis post-hoc descriptivo del modelo BP_FRONT PoseCNN ya cerrado. No se reentreno el modelo, no se ejecuto inferencia, no se modificaron probabilidades, threshold, split, preprocessing, labels, scaler ni checkpoint. Todas las conclusiones de rendimiento proceden exclusivamente de `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\bp_front_pose_test_predictions.csv` y del holdout Test congelado.

Las metricas globales permanecen sin cambios: Accuracy 0.8400, Precision 0.8548, Recall de la clase positiva valida 0.8833, F1 0.8689 y ROC-AUC 0.9354. La matriz es TN=31, FP=9, FN=7, TP=53. Clase positiva global: `1 = valido`.

El analisis por observabilidad explica resultados; no reemplaza estas metricas globales.

## 2. Informacion disponible en Pose

Input `(T,6,3)` con `x`, `y` y `confidence`:

| Index | Keypoint |
|---:|---|
| 0 | `left_elbow` |
| 1 | `left_shoulder` |
| 2 | `right_shoulder` |
| 3 | `right_elbow` |
| 4 | `right_wrist` |
| 5 | `left_wrist` |

Pose observa geometria y dinamica temporal de hombros, codos y munecas. No observa directamente barra, pecho, abdomen, cinturon, banco, cabeza, gluteos ni una mascara corporal completa. Que el modelo clasifique correctamente una muestra no demuestra que haya detectado directamente su falta reglamentaria.

## 3. Auditoria de codigos y criterios

El CSV original contiene 663 filas y cuatro codigos reales: `BP-01, BP-02, BP-03, BP-08`. Hay 29 muestras invalidas multicodigo y 9 invalidas sin codigo. `UNSPECIFIED` es un bucket descriptivo para estas ultimas, no un codigo reglamentario inventado.

| fault_code | criterion_description | pose_observability | available_pose_information | missing_information | justification | potentially_informative_modality |
|---|---|---|---|---|---|---|
| BP-01 | La barra no desciende al pecho/abdomen o toca el cinturón | DEBILMENTE_OBSERVABLE_POSE | Trayectoria temporal de ambas munecas y geometria relativa de munecas, codos y hombros como proxy de la trayectoria de la barra. | Barra, pecho, abdomen, cinturon y evidencia de contacto entre la barra y la superficie corporal reglamentaria. | Las munecas pueden correlacionarse con la altura de la barra, pero Pose no observa la barra ni el objetivo de contacto. Una clasificacion correcta solo demuestra uso de proxies o correlaciones del dataset. | POSE + SEGMENTATION |
| BP-02 | Los codos no bajan al nivel o por debajo de la superficie superior de los hombros | DIRECTAMENTE_OBSERVABLE_POSE | Posiciones x/y temporales de ambos hombros y ambos codos, que son las articulaciones directamente implicadas por el criterio. | Superficie anatomica completa de los hombros; los keypoints son puntos y no contornos corporales. | La relacion vertical codo-hombro puede medirse directamente con los keypoints disponibles. La falta de contorno introduce una aproximacion, pero no elimina la evidencia geometrica principal. | POSE |
| BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Trayectoria temporal bilateral de munecas, codos y hombros; las munecas suelen desplazarse junto con la barra durante el press. | Barra observada directamente y delimitacion explicita de la fase de press o del comando arbitral. | Una bajada bilateral de munecas puede ser evidencia parcial de movimiento descendente, pero Pose no prueba que la barra descendio ni identifica por si sola la fase reglamentaria exacta. | POSE + SEGMENTATION |
| BP-08 | Cabeza, hombros o glúteos se levantan del banco | DEBILMENTE_OBSERVABLE_POSE | Posicion temporal de ambos hombros y movimiento relativo de brazos. | Cabeza, gluteos, banco y contacto de cabeza/hombros/gluteos con el banco. | Solo los hombros estan representados y ni siquiera se observa la superficie del banco. El criterio disyuntivo tambien puede violarse por cabeza o gluteos, completamente ausentes del input. | POSE + SEGMENTATION |
| UNSPECIFIED | Codigo y criterio no informados en el CSV | INDETERMINADO | Secuencia general de hombros, codos y munecas. | Codigo y criterio de nulidad de la muestra. | Sin causa etiquetada no es posible relacionar el resultado con la informacion observable por Pose. | INFORMACION NO DISPONIBLE AUN |

La taxonomia se definio desde los criterios y los inputs antes de calcular resultados. `BP-02` es directamente observable; `BP-03` parcialmente observable; `BP-01` y `BP-08` debilmente observables. No hay un codigo real clasificado como `NO_OBSERVABLE_DIRECTAMENTE_POSE` en este CSV.

## 4. Frecuencia en dataset completo y Test

| fault_code | source_type | n_full_dataset | n_test |
|---|---|---|---|
| BP-01 | real_fault_code | 152 | 18 |
| BP-02 | real_fault_code | 87 | 14 |
| BP-03 | real_fault_code | 38 | 8 |
| BP-08 | real_fault_code | 6 | 1 |
| UNSPECIFIED | synthetic_bucket_for_missing_code | 9 | 1 |

Las frecuencias por codigo no suman el total de nulos porque una muestra multicodigo pertenece a cada uno de sus codigos. Las frecuencias del dataset completo son descriptivas; no se usaron predicciones de Train o Validation.

## 5. Null Detection Rate por codigo

Para una muestra realmente invalida, `predicted_label=0` es correcta y `predicted_label=1` es un falso positivo global. Se define:

`Null Detection Rate = invalidas del codigo predichas como invalidas / invalidas Test del codigo`.

No se denomina Recall porque el Recall global usa como clase positiva `1 = valido`.

| fault_code | pose_observability | n_test | n_correct_invalid | n_incorrect_valid | null_detection_rate | mean_probability_valid | median_probability_valid | min_probability_valid | max_probability_valid |
|---|---|---|---|---|---|---|---|---|---|
| BP-01 | DEBILMENTE_OBSERVABLE_POSE | 18 | 17 | 1 | 0.9444 | 0.0811 | 0.0018 | 0.0000 | 0.7320 |
| BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | 14 | 14 | 0 | 1.0000 | 0.0098 | 0.0011 | 0.0000 | 0.0948 |
| BP-03 | PARCIALMENTE_OBSERVABLE_POSE | 8 | 2 | 6 | 0.2500 | 0.6466 | 0.6402 | 0.0460 | 0.9836 |
| BP-08 | DEBILMENTE_OBSERVABLE_POSE | 1 | 0 | 1 | 0.0000 | 0.6053 | 0.6053 | 0.6053 | 0.6053 |
| UNSPECIFIED | INDETERMINADO | 1 | 0 | 1 | 0.0000 | 0.8970 | 0.8970 | 0.8970 | 0.8970 |

Resultados principales:

- `BP-02`, directamente observable, obtuvo 14/14 y Null Detection Rate 100.00%.
- `BP-03`, parcialmente observable, obtuvo 2/8 y 25.00%; seis de ocho fueron clasificados como validos.
- `BP-01`, debilmente observable, obtuvo 17/18 y 94.44%. Este resultado alto no prueba observacion de contacto barra-cuerpo; es compatible con proxies de munecas o correlaciones del dataset.
- `BP-08` tiene solo 1 caso Test y no permite una conclusion estable; fue clasificado como valido.

## 6. Analisis multi-hot por observabilidad

| observability_category | sample_type | membership_mode | n | correct_invalid | incorrect_as_valid | null_detection_rate | mean_probability_valid | median_probability_valid | correct_as_valid | incorrect_as_invalid | valid_classification_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DIRECTAMENTE_OBSERVABLE_POSE | INVALID | multi_hot_nonexclusive | 14 | 14.0000 | 0.0000 | 1.0000 | 0.0098 | 0.0011 | N/D | N/D | N/D |
| PARCIALMENTE_OBSERVABLE_POSE | INVALID | multi_hot_nonexclusive | 8 | 2.0000 | 6.0000 | 0.2500 | 0.6466 | 0.6402 | N/D | N/D | N/D |
| DEBILMENTE_OBSERVABLE_POSE | INVALID | multi_hot_nonexclusive | 19 | 17.0000 | 2.0000 | 0.8947 | 0.1087 | 0.0019 | N/D | N/D | N/D |
| NO_OBSERVABLE_DIRECTAMENTE_POSE | INVALID | multi_hot_nonexclusive | 0 | 0.0000 | 0.0000 | N/D | N/D | N/D | N/D | N/D | N/D |
| INDETERMINADO | INVALID | multi_hot_nonexclusive | 1 | 0.0000 | 1.0000 | 0.0000 | 0.8970 | 0.8970 | N/D | N/D | N/D |
| VALID | VALID | mutually_exclusive_valid_class | 60 | N/D | N/D | N/D | 0.8639 | 0.9902 | 53.0000 | 7.0000 | 0.8833 |

Las categorias invalidas son multi-hot y no mutuamente excluyentes. Una muestra con codigos de distinta observabilidad participa en todas las categorias correspondientes. Por eso sus N tampoco tienen que sumar 40. No se forzo una categoria unica por muestra.

## 7. Nueve falsos positivos

Estos son los nueve levantamientos invalidos clasificados como validos:

| video_id | true_label | predicted_label | probability_valid | fault_codes | criteria | pose_observability_categories | classification_interpretation |
|---|---|---|---|---|---|---|---|
| bp_053 | 0 | 1 | 0.5909 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_356 | 0 | 1 | 0.6895 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_357 | 0 | 1 | 0.5467 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_389 | 0 | 1 | 0.7320 | BP-01 | La barra no desciende al pecho/abdomen o toca el cinturón | DEBILMENTE_OBSERVABLE_POSE | Clasificacion incorrecta ante evidencia debil; la informacion reglamentaria ausente es una limitacion plausible, pero no prueba la causa individual del error. |
| bp_421 | 0 | 1 | 0.6053 | BP-08 | Cabeza, hombros o glúteos se levantan del banco | DEBILMENTE_OBSERVABLE_POSE | Clasificacion incorrecta ante evidencia debil; la informacion reglamentaria ausente es una limitacion plausible, pero no prueba la causa individual del error. |
| bp_471 | 0 | 1 | 0.9161 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_570 | 0 | 1 | 0.9836 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_571 | 0 | 1 | 0.9587 | BP-03 | Movimiento descendente de la barra durante el press | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion incorrecta con evidencia solo parcial; faltan barra y/o fase reglamentaria, aunque tambien puede existir error de aprendizaje del clasificador. |
| bp_606 | 0 | 1 | 0.8970 | UNSPECIFIED |  | INDETERMINADO | Interpretacion indeterminada porque la muestra no tiene codigo ni criterio de nulidad. |

Seis de los nueve FP contienen `BP-03`, uno `BP-01`, uno `BP-08` y uno no tiene codigo (`UNSPECIFIED`). No hay un FP con `BP-02` en este holdout. La baja observabilidad es una limitacion plausible para BP-01/BP-03/BP-08, pero no se atribuye automaticamente cada error a esa causa; tambien puede existir fallo de aprendizaje o sesgo de muestra. `bp_606` no puede interpretarse causalmente porque carece de codigo.

## 8. Treinta y un true negatives

Estos son los 31 nulos correctamente clasificados:

| video_id | probability_valid | fault_codes | pose_observability_categories | classification_interpretation |
|---|---|---|---|---|
| bp_261 | 0.0002 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_262 | 0.0000 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_278 | 0.0024 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_294 | 0.0001 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_298 | 0.0000 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_303 | 0.0000 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_304 | 0.0000 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_339 | 0.0042 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_342 | 0.0013 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_383 | 0.0000 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_387 | 0.4412 | BP-03 | PARCIALMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia parcial; puede apoyarse en cinemática de munecas, pero no prueba observacion directa de la barra o de la condicion reglamentaria. |
| bp_444 | 0.0770 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_454 | 0.0007 | BP-01\|BP-02 | DIRECTAMENTE_OBSERVABLE_POSE\|DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_470 | 0.2188 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_501 | 0.0019 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_503 | 0.0016 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_509 | 0.0460 | BP-01\|BP-03 | PARCIALMENTE_OBSERVABLE_POSE\|DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia parcial; puede apoyarse en cinemática de munecas, pero no prueba observacion directa de la barra o de la condicion reglamentaria. |
| bp_518 | 0.0041 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_529 | 0.3688 | BP-01 | DEBILMENTE_OBSERVABLE_POSE | Clasificacion correcta pese a evidencia debil; debe interpretarse como uso de proxies o correlaciones, no como deteccion directa de la falta. |
| bp_574 | 0.0948 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_590 | 0.0015 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_591 | 0.0052 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_595 | 0.0051 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_598 | 0.0034 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_602 | 0.0033 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_607 | 0.0226 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_628 | 0.0004 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_633 | 0.0006 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_643 | 0.0000 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_650 | 0.0001 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |
| bp_654 | 0.0001 | BP-02 | DIRECTAMENTE_OBSERVABLE_POSE | Clasificacion correcta con evidencia articular directamente relevante disponible en Pose. |

Los TN incluyen faltas directamente observables y faltas debil o parcialmente observables. Para estas ultimas debe decirse que el modelo clasifico correctamente la muestra mediante informacion indirecta potencial; no que detecto directamente la regla.

## 9. Casos multicodigo

En el Test hay 2 muestras multicodigo:

| video_id | fault_codes | pose_observability_categories | predicted_label | probability_valid | correct |
|---|---|---|---|---|---|
| bp_454 | BP-01\|BP-02 | DIRECTAMENTE_OBSERVABLE_POSE\|DEBILMENTE_OBSERVABLE_POSE | 0 | 0.0007 | True |
| bp_509 | BP-01\|BP-03 | PARCIALMENTE_OBSERVABLE_POSE\|DEBILMENTE_OBSERVABLE_POSE | 0 | 0.0460 | True |

Cada una participa en todas sus filas per-code y en todos sus flags multi-hot. No se reconciliaron ni priorizaron codigos.

## 10. Levantamientos validos

Los 60 validos reales se mantienen como categoria `VALID`: 53 fueron clasificados correctamente como validos (TP) y 7 como invalidos (FN). Su tasa de clasificacion valida es 88.33%. No se les asigna observabilidad de falta porque no tienen una causa de nulidad que explicar.

## 11. Proxies e interpretacion

PoseCNN puede usar trayectoria bilateral de munecas como proxy de la barra, angulos y relaciones codo-hombro, amplitud del descenso, sincronizacion bilateral, duracion relativa y patrones generales de ejecucion. Tambien puede explotar correlaciones no causales del dataset. En faltas sin evidencia reglamentaria completa, un acierto solo significa `clasifico correctamente`; no demuestra deteccion directa del codigo.

Los datos apoyan parcialmente la hipotesis de que Pose funciona mejor cuando la evidencia esta representada: BP-02 alcanza 100% y BP-03 solo 25%. Sin embargo, BP-01 tambien alcanza 94.44% pese a ser debilmente observable, por lo que no existe una relacion monotona simple entre observabilidad y rendimiento. Ese resultado refuerza la posibilidad de proxies o correlaciones. BP-08 tiene N=1 y no permite inferencia general.

## 12. Complementariedad esperada con Segmentation

| Codigo | Evidencia potencial principal | Interpretacion |
|---|---|---|
| BP-01 | Pose + Segmentation | Pose aporta munecas/brazos; segmentacion podria aportar barra y contorno corporal. No garantiza por si sola contacto semantico con pecho, abdomen o cinturon. |
| BP-02 | Pose | Hombros y codos contienen la relacion geometrica principal; una mascara podria refinar superficies. |
| BP-03 | Pose + Segmentation | Pose aporta cinemática articular y segmentacion podria aportar trayectoria directa de barra; aun debe identificarse la fase de press. |
| BP-08 | Pose + Segmentation | Pose aporta hombros y una mascara persona-banco podria aportar contacto global; cabeza y gluteos pueden seguir siendo ambiguos. |
| UNSPECIFIED | Informacion no disponible aun | Sin causa de nulidad no puede decidirse que modalidad aporta evidencia. |

No se afirma que Segmentation resuelva todos los criterios. La futura fusion esta metodologicamente justificada porque las modalidades aportan evidencia complementaria, pero debera evaluarse sobre el split common congelado y con sus propias limitaciones.

## 13. Limitaciones

- Los subconjuntos por codigo son pequenos y solapados; BP-08 tiene N=1.
- La observabilidad es una clasificacion tecnica basada en inputs y reglas, no una variable medida experimentalmente.
- No hay atribucion causal de features ni analisis de explicabilidad del modelo.
- Los resultados per-code no son metricas multiclase: PoseCNN sigue siendo binaria.
- Nueve invalidas originales no tienen codigo; una aparece en Test.
- No se analizaron predicciones de Train/Validation para generalizacion.

## 14. Trazabilidad

- Predicciones congeladas: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\bp_front_pose_test_predictions.csv`.
- Split congelado: `C:\Users\basti\MediapipePythonProjects\training\data\bp\front\bp_front_pose_split.csv`.
- Labels: `C:\Users\basti\MediapipePythonProjects\etiquetado\bp\bp_front.csv`.
- Analisis por codigo: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\analysis\bp_front_pose_test_by_fault_code.csv`.
- Nulos por muestra: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\analysis\bp_front_pose_test_invalid_samples.csv`.
- Taxonomia: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\analysis\bp_front_pose_fault_observability.csv`.
- Resumen de observabilidad: `C:\Users\basti\MediapipePythonProjects\training\results\bp\front\posecnn\analysis\bp_front_pose_observability_summary.csv`.

Este analisis no entreno PoseCNN, Segmentation CNN ni Fusion y no ejecuto inferencia.
