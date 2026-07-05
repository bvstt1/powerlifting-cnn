import cv2
import pandas as pd
import os
import re

# ----------------------------------
# CONFIGURACION
# ----------------------------------
RUTA_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bp_press_lateral_command.csv"
)

COLUMNAS_CSV = ["video_id", "press_frame"]

SALTO_FRAMES = 5       # frames para flechas izquierda/derecha
FPS_REPRODUCCION = 30  # velocidad de reproduccion al pausar
TAMANO_VENTANA = (960, 700)
NOMBRE_VENTANA = "Presione ESPACIO en el momento del comando Press"

# Raiz del proyecto (sube 2 niveles desde display/bp/)
RAIZ_PROYECTO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."
))

# ----------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------

def extraer_video_id(ruta_video):
    """
    Extrae el identificador del video a partir del nombre del archivo.
    Ejemplo: 'bp_001.mp4' -> 'bp_001'
    """
    nombre = os.path.splitext(os.path.basename(ruta_video))[0]
    return nombre


def cargar_csv_existente(ruta_csv):
    """
    Carga el CSV si existe y retorna un DataFrame.
    Si no existe, retorna un DataFrame vacio con las columnas definidas.
    """
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv, dtype={COLUMNAS_CSV[0]: str})
        return df
    return pd.DataFrame(columns=COLUMNAS_CSV)


def video_ya_etiquetado(video_id, df):
    """Verifica si el video ya fue etiquetado en el CSV."""
    return video_id in df[COLUMNAS_CSV[0]].values


def guardar_registro_csv(video_id, press_frame, ruta_csv):
    """
    Agrega o actualiza un registro en el CSV.
    Si el video ya existe, actualiza su press_frame.
    Si no existe, agrega una nueva fila.
    """
    df = cargar_csv_existente(ruta_csv)

    if video_id in df[COLUMNAS_CSV[0]].values:
        df.loc[df[COLUMNAS_CSV[0]] == video_id, COLUMNAS_CSV[1]] = press_frame
        print(f"Actualizado: {video_id} -> frame {press_frame}")
    else:
        nuevo = pd.DataFrame([[video_id, press_frame]], columns=COLUMNAS_CSV)
        df = pd.concat([df, nuevo], ignore_index=True)
        print(f"Guardado: {video_id} -> frame {press_frame}")

    df.to_csv(ruta_csv, index=False)


def mostrar_ayuda(frame, mensajes):
    """Superpone instrucciones en el frame de video."""
    h, w = frame.shape[:2]
    y0 = 30
    cv2.rectangle(frame, (0, 0), (w, y0 * len(mensajes) + 10), (0, 0, 0), -1)
    for i, msg in enumerate(mensajes):
        y = y0 * (i + 1)
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


RE_VIDEO_NUM = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


def generar_siguiente_ruta(ruta_video):
    """
    Dada una ruta como '.../bp_001.mp4', genera '.../bp_002.mp4'.
    Si el patron no coincide o el archivo no existe, retorna la misma ruta.
    """
    dirname = os.path.dirname(ruta_video)
    nombre = os.path.basename(ruta_video)
    m = RE_VIDEO_NUM.match(nombre)
    if not m:
        return ruta_video

    prefijo, num_str, ext = m.groups()
    nuevo_num = str(int(num_str) + 1).zfill(len(num_str))
    nueva_ruta = os.path.join(dirname, f"{prefijo}{nuevo_num}{ext}")

    # Si no existe, devolver la original
    return nueva_ruta if os.path.exists(nueva_ruta) else ruta_video


def seleccionar_video(ultima_ruta=""):
    """
    Solicita al usuario la ruta del video por consola.
    Si ultima_ruta esta definida (ej. el siguiente correlativo),
    se muestra como sugerencia al presionar Enter.
    Busca el archivo en varias ubicaciones para ser tolerante
    al directorio desde donde se ejecuta el script.
    Retorna la ruta absoluta si el archivo existe, o None si no.
    """
    prompt = "Ruta del video"
    if ultima_ruta:
        prompt += f" (Enter para: {os.path.basename(ultima_ruta)})"
    prompt += ": "

    ruta = input(prompt).strip().strip('"').strip("'")

    if not ruta and ultima_ruta:
        ruta = ultima_ruta

    if not ruta:
        print("No se ingreso ninguna ruta.")
        return None

    # Posibles resoluciones de la ruta
    candidatos = [
        os.path.abspath(ruta),
        os.path.abspath(os.path.join(RAIZ_PROYECTO, ruta)),
    ]

    for r in candidatos:
        if os.path.exists(r):
            return r

    print(f"ERROR: No se encuentra el archivo. Busque en:")
    for r in candidatos:
        print(f"  - {r}")
    return None


# ----------------------------------
# FUNCION PRINCIPAL DE REPRODUCCION
# ----------------------------------

def crear_ventana():
    """Crea la ventana OpenCV y la pone siempre al frente."""
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass  # fallback si el backend no soporta TOPMOST


def pantalla_inicio(cap, video_id, total_frames, fps):
    """
    Muestra el primer frame con instrucciones y espera ESPACIO para empezar.
    Esto le da tiempo al usuario de enfocar la ventana antes de la reproduccion.
    """
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return 0

        frame_mostrar = cv2.resize(frame, TAMANO_VENTANA)
        h, w = frame_mostrar.shape[:2]

        # Fondo semitransparente para el texto central
        overlay = frame_mostrar.copy()
        cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_mostrar, 0.4, 0, frame_mostrar)

        cv2.putText(frame_mostrar, "PRESIONE ESPACIO", (w // 2 - 180, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_mostrar, "para comenzar la reproduccion", (w // 2 - 220, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        mostrar_ayuda(frame_mostrar, [
            f"Video: {video_id} | Frames: {total_frames} | FPS: {fps:.1f}"
        ])

        cv2.imshow(NOMBRE_VENTANA, frame_mostrar)
        key = cv2.waitKey(30)

        if key == 32:  # ESPACIO -> comenzar
            return 0
        if key == 27:  # ESC -> cancelar
            return None


def reproducir_y_etiquetar(ruta_video):
    """
    Abre el video, reproduce y permite etiquetar el frame del comando Press.
    Al finalizar guarda automaticamente en CSV (con -1 si no se marco),
    y ofrece repetir el video o salir.

    Retorna True si el usuario quiere continuar con otro video,
    False si quiere salir del programa.

    Controles:
      ESPACIO  -> marcar press_frame
      P        -> pausar/reanudar
      <- / ->  -> retroceder/avanzar 5 frames (mas preciso si se mantiene)
      R        -> reiniciar la marca actual
      ESC      -> salir sin guardar
    """
    video_id = extraer_video_id(ruta_video)

    # Bucle exterior que permite repetir el video
    while True:

        cap = cv2.VideoCapture(ruta_video)

        if not cap.isOpened():
            print(f"ERROR: No se pudo abrir el video: {ruta_video}")
            return True

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_REPRODUCCION

        press_frame = None
        pausado = False
        frame_actual = 0

        # Cargar CSV existente para verificar si el video ya fue etiquetado
        df_csv = cargar_csv_existente(RUTA_CSV)
        if video_ya_etiquetado(video_id, df_csv):
            prev = df_csv.loc[df_csv[COLUMNAS_CSV[0]] == video_id, COLUMNAS_CSV[1]].values[0]
            print(f"Video ya etiquetado previamente: press_frame = {prev}")

        print(f"\n--- Etiquetando: {video_id} ---")
        print(f"Total frames: {total_frames} | FPS: {fps:.2f}")
        print("Controles: ESPACIO=Marcar  P=Pausa  <-/->=Ajustar  R=Reiniciar  ESC=Salir")

        # Ventana siempre al frente + pantalla de inicio para tomar foco
        crear_ventana()
        inicio = pantalla_inicio(cap, video_id, total_frames, fps)
        if inicio is None:
            cap.release()
            cv2.destroyAllWindows()
            print("Etiquetado cancelado.")
            return True
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Bucle interno de reproduccion
        ultimo_frame = None
        while True:
            if not pausado:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
                ret, frame = cap.read()
                if not ret:
                    break

            if frame is None:
                break

            ultimo_frame = frame.copy()
            h, w = frame.shape[:2]

            mensajes = [
                f"Video: {video_id}",
                f"Frame: {frame_actual + 1} / {total_frames}",
            ]

            if press_frame is not None:
                mensajes.append(f"Press marcado: frame {press_frame + 1}  [R=reiniciar]")
                cv2.circle(frame, (w - 50, 50), 15, (0, 200, 0), -1)
                cv2.circle(frame, (w - 50, 50), 15, (255, 255, 255), 2)
                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 200, 0), 1)
            else:
                mensajes.append("Press: NO MARCADO  [ESPACIO=Marcar]")

            mensajes.append("[P] Pausa  [<-/->] Ajustar  [ESC] Salir")

            mostrar_ayuda(frame, mensajes)

            frame_display = cv2.resize(frame, TAMANO_VENTANA)
            cv2.imshow(NOMBRE_VENTANA, frame_display)

            # NOTA: No usar & 0xFF porque las flechas en Windows
            # retornan valores multi-byte (2424832, 2555904, etc.)
            key = cv2.waitKey(int(1000 / fps) if not pausado else 50)

            if key == 27:  # ESC
                break

            elif key == 32:  # ESPACIO -> marcar press_frame
                press_frame = frame_actual
                print(f"MARCAR: press_frame = {frame_actual + 1}")

            elif key == ord('p') or key == ord('P'):
                pausado = not pausado

            elif key == ord('r') or key == ord('R'):
                if press_frame is not None:
                    press_frame = None

            # Flechas: 81/83 (Linux), 2424832/2555904 (Windows 64-bit)
            elif key in (81, 2424832):  # <- retroceder
                if not pausado:
                    pausado = True

                frame_actual = max(0, frame_actual - SALTO_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)

            elif key in (83, 2555904):  # -> avanzar
                if not pausado:
                    pausado = True

                frame_actual = min(total_frames - 1, frame_actual + SALTO_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)

        # ---- Menu final en la ventana de video ----
        valor_guardar = press_frame if press_frame is not None else -1
        guardar_registro_csv(video_id, valor_guardar, RUTA_CSV)

        decision = None
        while decision is None and ultimo_frame is not None:
            menu = cv2.resize(ultimo_frame.copy(), TAMANO_VENTANA)
            h, w = menu.shape[:2]

            overlay = menu.copy()
            cv2.rectangle(overlay, (w // 5, h // 4), (4 * w // 5, 3 * h // 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, menu, 0.35, 0, menu)

            titulo = "VIDEO FINALIZADO"
            if press_frame is not None:
                subt = f"Press frame: {press_frame + 1}"
            else:
                subt = "Press frame: NO MARCADO"
            cv2.putText(menu, titulo, (w // 2 - 140, h // 2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(menu, subt, (w // 2 - 120, h // 2 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(menu, "[R] Repetir video", (w // 2 - 100, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(menu, "[Enter] Siguiente video", (w // 2 - 100, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(menu, "[N] Salir del programa", (w // 2 - 100, h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(NOMBRE_VENTANA, menu)
            k = cv2.waitKey(50)

            if k in (ord('r'), ord('R')):
                decision = "repetir"
            elif k == 13 or k == 32:  # Enter o SPACE
                decision = "siguiente"
            elif k in (ord('n'), ord('N')):
                decision = "salir"

        cap.release()
        cv2.destroyAllWindows()

        if decision == "repetir":
            continue
        elif decision == "salir":
            return False
        else:
            return True


# ----------------------------------
# PUNTO DE ENTRADA
# ----------------------------------

def main():
    """
    Bucle principal: permite etiquetar multiples videos en una sesion.
    """
    print("=" * 60)
    print("HERRAMIENTA DE ETIQUETADO - PRESS FRAME")
    print("Marca el frame exacto donde el juez da el comando 'Press'")
    print("=" * 60)

    primera_vez = True
    while True:
        if primera_vez:
            ruta_video = seleccionar_video("")
            if ruta_video is None:
                print("Saliendo.")
                break
            primera_vez = False
        else:
            if not os.path.exists(ultima_ruta):
                print(f"\nNo existe el siguiente video: {ultima_ruta}")
                ruta_video = seleccionar_video(ultima_ruta)
                if ruta_video is None:
                    break
            else:
                ruta_video = ultima_ruta
                print(f"\nSiguiente video: {ruta_video}")

        continuar = reproducir_y_etiquetar(ruta_video)
        if not continuar:
            break
        ultima_ruta = generar_siguiente_ruta(ruta_video)

    print("\nHerramienta finalizada. CSV guardado en:")
    print(f"  {RUTA_CSV}")


if __name__ == "__main__":
    main()
