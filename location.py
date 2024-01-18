import json 

location={
    0:[[482,500],[893,502],[560,329],[773,330]],
    1:[[452,499],[882,501],[549,306],[780,305]],
    2:[[415,475],[862,472],[526,279],[762,281]],
    3:[[415,480],[822,470],[524,305],[736,305]]
    }
jsonfile='./location.json'
with open(jsonfile,'w') as f:
    json.dump(location,f)

with open(jsonfile,'r') as f:
    location1=json.load(f)

print(location1['1'])