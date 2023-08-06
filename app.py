from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/process", methods = ["POST", "GET"])
def process_prompt():
    # input_text = request.form["prompt"]
    # print(input_text)
    form_items = list(request.form.items())
    return f'Form items: {form_items}'
    return redirect(url_for("new.html", answer=input_text))
    # return render_template("new.html")
    form_items = list(request.form.items())
    return f'Form items: {form_items}'




if __name__ == "__main__":
    app.run(debug="True", port = 9599)    