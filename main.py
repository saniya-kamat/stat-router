import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Honour the PORT env var if set (e.g. by tooling); default to 5000 so the
    # usual `python main.py` keeps serving on the documented port.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)