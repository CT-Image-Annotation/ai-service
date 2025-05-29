from flask import Flask
from medsam2 import bp as ai_bp
from MEDSAM3D import bp as vd_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(ai_bp)
    app.register_blueprint(vd_bp)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
