from django.shortcuts import render
from .utils import predicted_class

def home(request):
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def predict(request):
    prediction = None
    image_data = None
    example_image_data = None

    if request.method == "POST":
        prediction,image_data,example_image_data = predicted_class(request)
    return render(request, "predict.html", {
        "prediction": prediction,
        "image_data": image_data,
        "example_image_data": example_image_data
    })