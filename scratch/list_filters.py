import pymeshlab
ms = pymeshlab.MeshSet()
# print all methods of MeshSet
for method in dir(ms):
    if "apply" in method or "filter" in method:
        print(method)
