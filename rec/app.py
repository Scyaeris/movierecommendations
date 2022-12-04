#Import modules
from flask import Flask, redirect, render_template, request, flash, url_for
import knn_model
import http.client
import random
#Initiate flask app and integrate API source key
app = Flask(__name__)

#Define route for home index
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/knn")
def knn():
    return render_template("knn.html")

@app.route("/nn")
def nn():
    flash("Search to begin.")
    return render_template("nn.html")

@app.route("/predictKNN", methods = ['POST'])
def predictKNN():

    data = request.form.get("data")
    x = data.find("/")

    if len(data) == 0:
        flash("You can not input a blank string.", 'error')
        return redirect(url_for('knn'))
    elif "/" not in data:
        result = knn_model.recommend(data, knn_model.getCSR(), knn_model.getModel(), knn_model.getMovieToIndex(), 5)
        if result == None:
            flash("This movie does not exist within the dataset. Please try a different movie.", 'error')
            return redirect(url_for('knn'))
    elif "/" in data and data[x + 1:].isdigit():
        title = data[:x]
        n = data[x + 1:]
        x = int(n)
        if x > 10:
            flash("Your query size was too large. Defaulting to size 5.", 'warning')
            result = knn_model.recommend(title, knn_model.getCSR(), knn_model.getModel(), knn_model.getMovieToIndex(), 5)
            return render_template('knn.html', result = result)
        else:
            result = knn_model.recommend(title, knn_model.getCSR(), knn_model.getModel(), knn_model.getMovieToIndex(), int(n))
        if result == None:
            flash("This movie does not exist within the dataset. Please try a different movie.", 'error')
            return redirect(url_for('knn'))
    else:
        flash("Something went wrong. Please try again.", 'error')
        return redirect(url_for('knn'))

    return render_template('knn.html', result = result)

@app.route("/predictNN", methods = ['POST'])
def predictNN():

    data = request.form.get("data")

    if len(data) == 0:
        flash("You can not input a blank string.")
        return render_template('nn.html')
    elif len(data) > 0:
        result = knn_model.recommend(data, knn_model.getCSR(), knn_model.getModel(), knn_model.getMovieToIndex(), 12)
        if result == None:
            flash("This movie does not exist within the dataset.")
            return render_template('nn.html')
    else:
        flash("Something went wrong. Please try again.")
        return render_template('nn.html')

    image_urls = []
    randInt = []

    for item in result:
        item_parse = item.replace(" ", '')
        item_parse = item_parse.encode()
        input_parsed = str(item_parse).lstrip('b')
        conn = http.client.HTTPSConnection("bing-image-search1.p.rapidapi.com")
        headers = {
            'X-RapidAPI-Key': "7e0ff16003msh9fd9042dd24b37ep13cf04jsned3e016768fa",
            'X-RapidAPI-Host': "bing-image-search1.p.rapidapi.com"
        }
        conn.request("GET", "/images/search?q=" + input_parsed + "%20landscape%20imdb%20cover&count=1", headers=headers)
        
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        start = data.find("thumbnailUrl") + 16
        stop = data.find('pid=Api"') + 7
        url = data[start:stop]

        randInt.append(random.randint(2,9))
        image_urls.append(url)

    return render_template('nn.html', result = result, images = image_urls, random = randInt)

#Enable debug mode for development purposes
if __name__ == "__main__":
    app.run(debug=True)
app.secret_key = 'WDNAJHDKUAHJSKDJKWAJOIS'