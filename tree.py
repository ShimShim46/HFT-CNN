import copy
from collections import defaultdict

# 階層構造(木)の作成
# =========================================================
def make(): return defaultdict(make)


def dicts(t): return {k: dicts(t[k]) for k in t}

# 階層構造(木)へラベルを追加
# ========================================================= 
def add(t, path):
    for node in path:
        t = t[node]

# 親ラベルの探索
# =========================================================
def search_parent(tree,child,layer=1,prev_parent='root'):
    for k,v in list(tree.items()):
        if(k == child):
                return prev_parent
        else:
                if len(v) >= 1:
                        layer += 1
                        found = search_parent(v, child, layer,k)
                        if found:
                                return found
                        layer -=1
                else:
                        continue

# 子ラベルの探索
# =========================================================
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

# 指定ラベルの根からのパスを探索
# =========================================================
def search_path(tree, node):
        start_node = copy.deepcopy(node)
        path = [start_node]
        while (node != "root"):
                node = search_parent(tree, node)
                path.append(node)

        return path
