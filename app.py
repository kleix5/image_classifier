from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import os
import zipfile
from openVDoc import roi_matching
from openVDoc import roi_matching_ov

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = ["uploads/unpack_archive", "uploads/images"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return "There is no file in submitted form!"
        file = request.files["file"]
        zip_abs = os.path.join(app.config["UPLOAD_FOLDER"][0], file.filename)
        file2 = request.files["file2"]
        file.save(zip_abs)
        file2.save(os.path.join(app.config["UPLOAD_FOLDER"][1], file2.filename))
        with zipfile.ZipFile(zip_abs, "r") as zip_ref:
            zip_ref.extractall(app.config["UPLOAD_FOLDER"][0])
            zip_ref.close()
        os.remove(zip_abs)
        # os.remove(os.path.join(app.config["UPLOAD_FOLDER"][0], file.filename))
        if request.form["model"] == "1":
            result = roi_matching.run_match(os.path.join(app.config["UPLOAD_FOLDER"][1], file2.filename))
        else:
            result = roi_matching_ov.roi_match()

        return str(result)
    else:
        return "Invalid request!!!"


