import requests

url='http://127.0.0.1:5000/api/v1/predictor/diagnosis'
files={'image': open('../gaussian_filtered_images/gaussian_filtered_images/Mild/0a61bddab956.png','rb')}
values={'image' : '../gaussian_filtered_images/gaussian_filtered_images/Mild/0a61bddab956.png'}
r=requests.post(url,files=files,data=values)
print(r.json())