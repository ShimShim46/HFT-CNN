from collections import defaultdict
import copy

def make(): return defaultdict(make)

def dicts(t): return {k: dicts(t[k]) for k in t}
 
def add(t, path):
    for node in path:
        t = t[node]

def search_parent(tree,child,layer=1,prevParent='root'):
    for k,v in list(tree.items()):
        if(k == child):
                return prevParent
        else:
                if len(v) >= 1:
                        layer += 1
                        found = search_parent(v, child, layer,k)
                        if found:
                                return found
                        layer -=1
                else:
                        continue

def search_child(tree,node,layer=1):
    if (node == "root" or node =="ROOT" or node == "Root"):
        return list(tree.keys())
    for k,v in list(tree.items()):
        if(k == node):
            return list(v.keys())
        else:
            if len(v) >= 1:
                layer += 1
                found = search_child(v, node, layer)
                if found:
                    return found
                layer -=1
            else:
                continue

def search_path(tree, node):
        startNode = copy.deepcopy(node)
        path = [startNode]
        while (node != "root"):
                node = search_parent(tree, node)
                path.append(node)

        return path