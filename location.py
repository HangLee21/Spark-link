import json 

location={
    0:[[413,562],[889,538],[519,297],[788,307]],
    1:[[408,506],[866,519],[523,289],[778,292]],
    2:[[437,468],[895,480],[514,251],[781,247]],
    3:[[399,485],[867,477],[525,252],[777,252]]
    }
jsonfile='./location.json'
with open(jsonfile,'w') as f:
    json.dump(location,f)

with open(jsonfile,'r') as f:
    location1=json.load(f)

print(location1['1'])