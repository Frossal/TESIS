# launch_app.py
import os, sys, socket, threading, webbrowser, time, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

LOG_PATH = os.path.join(BASE_DIR, "launcher.log")

def log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def find_free_port(start=8501, limit=20):
    port = start
    for _ in range(limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return start  # fallback

try:
    port = int(os.environ.get("STREAMLIT_SERVER_PORT", "0")) or find_free_port()
    url = f"http://127.0.0.1:{port}"

    # Config recomendada
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")

    # Abrimos el navegador nosotros (Streamlit a veces no lo hace dentro del exe)
    def open_browser():
        # Espera breve para que el server arranque
        time.sleep(2.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # Lanza streamlit como módulo (NO headless para permitir abrir browser)
    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(BASE_DIR, "app.py"),
        "--server.port", str(port),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--server.address", "127.0.0.1",
    ]
    log(f"Iniciando Streamlit en {url}")
    sys.exit(stcli.main())

except Exception as e:
    # Si algo sale mal, lo dejamos en un log para que puedas leerlo
    tb = traceback.format_exc()
    log("ERROR de lanzamiento:\n" + tb)
    # Intenta mostrar una ventana simple (si el exe no es windowed)
    print("No se pudo lanzar la app. Revisa launcher.log para más detalles.")
    raise
