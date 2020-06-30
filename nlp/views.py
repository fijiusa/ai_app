from django.shortcuts import render
import pandas as pd
import pickle

category_data=pd.read_csv("idx2category.csv")
idx2category={row.k: row.v for idx, row in category_data.iterrows()}

with open("multi.pickle", mode="rb") as f:
    model=pickle.load(f)

# Create your views here.
from django.http import HttpResponse


def index(request):
    #methodのGETとは、アクセスされた時
    if request.method=="GET":
        return render (
            request,
            "nlp/home.html"
        )
    else:
        #htmlのnameに指定した場所を呼び出し値を取得。今回はinputタグ。
        title=[request.POST["title"]]
        print("title:", title)
        result=model.predict(title)[0]
        print("result:", result)
        pred = idx2category[result]
        return render (
                request,
                "nlp/home.html",
                {"title":pred}
        )

