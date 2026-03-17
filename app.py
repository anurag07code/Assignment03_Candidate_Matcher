import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from processor import build_candidate_matcher_index, run_candidate_matching


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
    app.config["RESUMES_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "resumes")
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB per request

    os.makedirs(app.config["RESUMES_FOLDER"], exist_ok=True)

    state = {"matcher": None}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "matcher_ready": state["matcher"] is not None})

    @app.post("/upload")
    def upload_resumes():
        files = request.files.getlist("files")
        if not files:
            return jsonify({"message": "No files uploaded."}), 400

        saved = 0
        for f in files:
            name = secure_filename(f.filename or "")
            if not name:
                continue
            if not name.lower().endswith(".pdf"):
                return jsonify({"message": "Only PDF resumes are supported in this section."}), 400
            out_path = os.path.join(app.config["RESUMES_FOLDER"], name)
            f.save(out_path)
            saved += 1

        if saved == 0:
            return jsonify({"message": "No valid PDF files were uploaded."}), 400

        return jsonify({"message": f"{saved} resumes uploaded. Ready to build matcher index."})

    @app.post("/matcher/build")
    def matcher_build():
        if not os.path.isdir(app.config["RESUMES_FOLDER"]) or not os.listdir(app.config["RESUMES_FOLDER"]):
            return jsonify({"message": "No resumes uploaded. Upload PDF resumes first."}), 400

        try:
            state["matcher"] = build_candidate_matcher_index(resumes_dir=app.config["RESUMES_FOLDER"])
        except Exception as e:
            return jsonify({"message": f"Failed to build matcher index: {str(e)}"}), 500

        return jsonify({"message": "Candidate matcher index built successfully!"})

    @app.post("/matcher/match")
    def matcher_match():
        if state["matcher"] is None:
            return jsonify({"message": "Build the matcher index first."}), 400

        data = request.get_json(silent=True) or {}
        job_description = (data.get("job_description") or "").strip()
        if not job_description:
            return jsonify({"message": "Job description is required to run matching."}), 400

        try:
            top_k_raw = data.get("top_k")
            top_k = int(top_k_raw) if top_k_raw is not None else 0
        except Exception:
            top_k = 0
        if top_k <= 0:
            top_k = len(state["matcher"].resumes)

        try:
            result = run_candidate_matching(state["matcher"], job_description=job_description, top_k=top_k)
        except Exception as e:
            return jsonify({"message": f"Failed to run matching: {str(e)}"}), 500

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

