import json 

location={
    0:[[448,506],[836,504],[512,301],[763,300]],
    1:[[474,516],[872,533],[558,306],[818,305]],
    2:[[456,429],[828,422],[517,239],[758,239]],
    3:[[446,434],[819,448],[526,249],[766,253]]
    }
jsonfile='./location.json'
with open(jsonfile,'w') as f:
    json.dump(location,f)

with open(jsonfile,'r') as f:
    location1=json.load(f)

print(location1['1'])